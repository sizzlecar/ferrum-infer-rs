//! The original sampler has already selected/validated its candidate. This
//! leaf never edits logits, masks, history, KV, completion or output directly.
use super::*;
use crate::continuous_engine::{advance_pending_utf8_fragment, SequenceSamplingHistoryScope};
use ferrum_interfaces::Tokenizer;

impl SequenceState {
    pub(in crate::continuous_engine) fn commit_selected_token_with_prefix(
        &mut self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        original: TokenId,
        route: PrefixCandidateRouteV1,
        processed_logits: Option<&[f32]>,
    ) -> Result<TokenId> {
        let Some(installed) = &self.calibration_prefix else {
            self.commit_generated_token(tokenizer, original)?;
            return Ok(original);
        };
        let tokenizer = tokenizer.ok_or_else(|| invalid("prefix tokenizer absent"))?;
        let frontier = self
            .cost_frontier
            .ok_or_else(|| invalid("prefix owner absent"))?;
        let declaration = installed.plan.declaration();
        if frontier.owner_incarnation.get() != installed.owner
            || installed.pending_commit.is_some()
            || self.generated_tokens.len() >= declaration.release_generated
            || self.generated_tokens.len() >= self.sampling_params.max_tokens
            || self.sampling_history.scope() != SequenceSamplingHistoryScope::FullGeneration
            || self.structured_output_processor.is_some()
            || !self.stop_token_ids.is_empty()
            || !self.model_eos_token_ids.is_empty()
            || !self.user_stop_token_ids.is_empty()
            || !self.stop_text_seqs.is_empty()
            || tokenizer.host_output_policy_identity() != Some(declaration.tokenizer_policy_sha256)
        {
            return Err(invalid(
                "prefix capability/frontier/policy differs from declaration",
            ));
        }
        let token = *declaration
            .token_ids
            .get(self.generated_tokens.len())
            .ok_or_else(|| invalid("prefix fixed frontier exhausted"))?;
        let id = token.get();
        let mask = if self.generated_tokens.is_empty() {
            self.initial_argmax_token_mask
                .as_ref()
                .or(self.argmax_token_mask.as_ref())
        } else {
            self.argmax_token_mask.as_ref()
        };
        if usize::from(token) >= tokenizer.vocab_size()
            || self.forbidden_token_ids.contains(&id)
            || (self.generated_tokens.is_empty() && self.initial_forbidden_token_ids.contains(&id))
            || (self
                .tokenizer_base_vocab_size
                .is_some_and(|base| usize::from(token) >= base)
                && !self.allowed_extended_token_ids.contains(&id))
            // The backend Greedy mask additionally excludes context-free
            // invalid byte tokens. Full logits use the original pending-aware
            // UTF-8 table and processed logits instead: a continuation byte
            // can be legal after an incomplete prefix while masked for Greedy.
            || (route == PrefixCandidateRouteV1::ModelGreedyArgmax && mask.is_some_and(|mask| {
                mask.valid_token_mask
                    .get(usize::from(token))
                    .is_none_or(|&v| v == 0)
            }))
            || processed_logits.is_some_and(|logits| {
                logits
                    .get(usize::from(token))
                    .is_none_or(|v| !v.is_finite())
            })
            || self.sample_candidate_decodes_to_forbidden_output(
                Some(tokenizer),
                self.decoded_text_len,
                token,
                None,
            )
        {
            return Err(invalid(
                "declared prefix token violates actual sampling/output constraints",
            ));
        }
        // commit_generated_token performs the same byte validation again at
        // the original mutation boundary; this read only binds the evidence.
        let bytes = self
            .token_bytes_for_output(tokenizer, token)?
            .ok_or_else(|| invalid("prefix raw token bytes unavailable"))?;
        let after = advance_pending_utf8_fragment(&self.pending_decoded_utf8_bytes, &bytes)
            .map_err(|()| invalid("prefix pending UTF-8 transition is invalid"))?;
        let event = PrefixTokenCommitV1 {
            request_id: self.request_id.clone(),
            owner_incarnation: frontier.owner_incarnation.get(),
            work_generation: frontier.work_generation.get(),
            generated_before: self.generated_tokens.len(),
            generated_after: self
                .generated_tokens
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("prefix generated frontier overflow"))?,
            original_candidate: original,
            committed_token: token,
            route,
            pending_before: self.pending_decoded_utf8_bytes.clone(),
            pending_after: after,
        };
        self.commit_generated_token(Some(tokenizer), token)?;
        // No fallible work or new allocation after actual commit.
        self.calibration_prefix
            .as_mut()
            .expect("private prefix remains installed")
            .pending_commit = Some(event);
        Ok(token)
    }
}
