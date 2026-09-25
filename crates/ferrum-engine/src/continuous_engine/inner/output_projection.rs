//! Immutable text work detached from global sequence-state ownership.
use super::*;
use crate::continuous_engine::{
    credited_output::CreditedDecodePolicy, output_flow_runtime::OutputProjectionGuard,
};

#[cfg(test)]
mod tests;

/// Only immutable text/protocol inputs are copied. No model resource, sampler,
/// structured-output processor, or response sender escapes in the snapshot.
pub(super) struct StreamProjectionSnapshot {
    incarnation: Arc<()>,
    tokens: Vec<TokenId>,
    stops: Vec<String>,
    protocol: ferrum_types::ModelOutputProtocol,
    last_token_is_stop: bool,
    streamed_text_len: usize,
    pending_utf8: [u8; 3],
    pending_utf8_len: usize,
    decoder: Option<CreditedDecodePolicy>,
    // Last: all detached token/stop storage must die before its lifetime guard.
    _projection_lifetime: Option<OutputProjectionGuard>,
}

impl StreamProjectionSnapshot {
    pub(super) fn capture(sequence: &SequenceState) -> Self {
        Self {
            incarnation: sequence.stream_projection_identity.clone(),
            tokens: sequence.generated_tokens.clone(),
            stops: sequence.stop_text_seqs.clone(),
            protocol: sequence.sampling_params.model_output_protocol,
            last_token_is_stop: sequence
                .generated_tokens
                .last()
                .is_some_and(|token| sequence.stop_token_ids.contains(&token.get())),
            streamed_text_len: sequence.streamed_text_len,
            pending_utf8: {
                let mut bytes = [0; 3];
                let len = sequence.pending_decoded_utf8_bytes.len().min(bytes.len());
                bytes[..len].copy_from_slice(&sequence.pending_decoded_utf8_bytes[..len]);
                bytes
            },
            pending_utf8_len: sequence.pending_decoded_utf8_bytes.len(),
            decoder: sequence
                .credited_output
                .as_ref()
                .map(|output| output.decoder),
            _projection_lifetime: sequence
                .credited_output
                .as_ref()
                .map(|output| output.port.projection_guard()),
        }
    }

    /// Keep full-history decoding: a token may complete an earlier UTF-8
    /// fragment or release a held stop prefix. A one-token decode is not equal.
    pub(super) fn project(
        self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        terminal: Option<FinishReason>,
    ) -> Result<Option<StreamTextProjection>> {
        let text = self.project_text(tokenizer, terminal)?;
        Ok(text.map(|text| StreamTextProjection {
            snapshot: self,
            text,
        }))
    }

    /// Keep the fence even when decode fails or yields no complete UTF-8.
    /// Those outcomes can change output ownership and therefore need the same
    /// incarnation/frontier protection as a successful text projection.
    pub(super) fn project_fenced(
        self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        terminal: Option<FinishReason>,
    ) -> FencedStreamTextProjection {
        let result = self.project_text(tokenizer, terminal);
        FencedStreamTextProjection {
            snapshot: self,
            result,
        }
    }

    fn project_text(
        &self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        terminal: Option<FinishReason>,
    ) -> Result<Option<ProjectedText>> {
        let mut visible = self.decode(tokenizer, &self.tokens)?;
        let mut terminal_utf8_proven = false;
        if matches!(
            terminal,
            Some(FinishReason::Length | FinishReason::Cancelled)
        ) {
            let proof_required = self.pending_utf8_len != 0 || visible.ends_with('\u{FFFD}');
            let proven = crate::continuous_engine::credited_output::finish_incomplete_utf8(
                tokenizer,
                &self.tokens,
                self.pending_utf8
                    .get(..self.pending_utf8_len)
                    .ok_or_else(|| {
                        FerrumError::internal("terminal UTF-8 pending proof exceeds three bytes")
                    })?,
                self.decoder,
                &mut visible,
            )?;
            if !proven {
                return Ok(None);
            }
            terminal_utf8_proven = proof_required;
        }
        let incomplete_utf8 = visible.ends_with('\u{FFFD}') && !terminal_utf8_proven;
        if incomplete_utf8 && terminal.is_none() {
            return Ok(None);
        }
        let decoded_text_len = (!incomplete_utf8).then_some(visible.len());
        let end = visible_text_end(&self.stops, &visible, terminal.is_some());
        if end < visible.len() {
            visible.truncate(end);
        } else if let (Some(reason), Some(&last)) = (terminal, self.tokens.last()) {
            // decoded_output_text only consults terminal visibility when the
            // last token is a stop ID. Its grammar-completion alternative
            // explicitly requires !stop_ids.contains(last), so cannot apply
            // here. Harmony's actual typed marker IDs remain observable.
            let keep_terminal = match reason {
                FinishReason::Error => false,
                FinishReason::Stop | FinishReason::EOS => {
                    self.protocol == ferrum_types::ModelOutputProtocol::HarmonyGptOss
                        && ["<|call|>", "<|return|>"]
                            .into_iter()
                            .filter_map(|marker| tokenizer.token_id(marker))
                            .any(|marker| marker == last)
                }
                _ => true,
            };
            if self.last_token_is_stop && !keep_terminal {
                visible = self.decode(tokenizer, &self.tokens[..self.tokens.len() - 1])?;
            }
        }
        if visible.ends_with('\u{FFFD}') && !terminal_utf8_proven {
            return Ok(None);
        }
        Ok(Some(ProjectedText {
            visible,
            decoded_text_len,
        }))
    }

    fn decode(
        &self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        tokens: &[TokenId],
    ) -> Result<String> {
        match self.decoder {
            Some(decoder) => decoder.decode(tokenizer, tokens),
            None => tokenizer.decode(tokens, true),
        }
    }
}

struct ProjectedText {
    visible: String,
    decoded_text_len: Option<usize>,
}

impl ProjectedText {
    fn commit(self, sequence: &mut SequenceState) -> Option<String> {
        let delta = self.visible.get(sequence.streamed_text_len..)?.to_owned();
        if let Some(length) = self.decoded_text_len {
            sequence.decoded_text_len = length;
        }
        sequence.streamed_text_len = self.visible.len();
        Some(delta)
    }
}

pub(super) struct FencedStreamTextProjection {
    result: Result<Option<ProjectedText>>,
    // Drop decoded text before the snapshot releases the projection lifetime.
    snapshot: StreamProjectionSnapshot,
}

impl FencedStreamTextProjection {
    /// None means stale work: it must neither consume the current request's
    /// grant nor alter its failure/timing/watermark state. The credited route
    /// owns a single grant, so every outcome requires the exact old watermark.
    pub(super) fn commit(self, sequence: &mut SequenceState) -> Option<Result<Option<String>>> {
        if !Arc::ptr_eq(
            &self.snapshot.incarnation,
            &sequence.stream_projection_identity,
        ) || self.snapshot.tokens != sequence.generated_tokens
            || self.snapshot.streamed_text_len != sequence.streamed_text_len
        {
            return None;
        }
        Some(self.result.and_then(|text| {
            text.map(|text| {
                text.commit(sequence).ok_or_else(|| {
                    FerrumError::internal("output projection rewrote an emitted text frontier")
                })
            })
            .transpose()
        }))
    }
}

pub(super) struct StreamTextProjection {
    text: ProjectedText,
    snapshot: StreamProjectionSnapshot,
}

impl StreamTextProjection {
    /// Revalidate the exact frontier, incarnation and watermark before changing
    /// output state. Comparison is required even when token counts are equal:
    /// rollback/replacement can change history without changing its length.
    pub(super) fn commit(self, sequence: &mut SequenceState) -> Option<String> {
        if !Arc::ptr_eq(
            &self.snapshot.incarnation,
            &sequence.stream_projection_identity,
        ) || self.snapshot.tokens != sequence.generated_tokens
            || sequence.streamed_text_len < self.snapshot.streamed_text_len
        {
            return None;
        }
        // Another projection of this exact frontier may already have emitted
        // a prefix (e.g. the nonterminal stop-prefix view). Rebase onto the
        // current validated watermark without repeating already visible text.
        self.text.commit(sequence)
    }
}

pub(super) fn visible_text_end(stops: &[String], text: &str, terminal: bool) -> usize {
    if let Some(end) = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
    {
        return end;
    }
    if terminal {
        return text.len();
    }
    let held = stops
        .iter()
        .flat_map(|stop| {
            stop.char_indices()
                .skip(1)
                .map(move |(len, _)| &stop[..len])
        })
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    text.len() - held
}
