//! Admission-only identity of the actual host sampling/output path.
//!
//! This is cached before the first wave. It intentionally excludes prompt and
//! request identities, random seed, clocks, cache addresses and SLO mode. A
//! declared tokenizer implementation/content identity is required; algorithm
//! names or memory bounds cannot identify an arbitrary trait implementation.

use crate::continuous_engine::SequenceState;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostCostPolicyV2};
use ferrum_interfaces::{sampler::SamplingConfig, Tokenizer};
use ferrum_types::ApiRequest;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, io};

// The tokenizer and sampling parameters do not identify the engine algorithm.
// Change this revision when host sampling/projection/terminal work changes;
// historical observations must not acquire a new algorithm's cost authority.
const HOST_OUTPUT_ALGORITHM_REVISION: &str =
    "utf8-pretruncation-fsm.v1/terminal-byte-proof.v1/unsubmitted-output-grant.v1";

/// Call only after installing the real legacy sender or credited owner, and
/// before the sequence starts executing. Unknown/mismatched policy sources
/// disable training instead of inventing a broadly reusable cost key.
pub(in crate::continuous_engine) fn host_policy_signature(
    sequence: &SequenceState,
    tokenizer: &dyn Tokenizer,
) -> Option<[u8; 32]> {
    policy_signature(sequence, tokenizer, false)
}

pub(in crate::continuous_engine) fn host_numeric_policy(
    sequence: &SequenceState,
    tokenizer: &dyn Tokenizer,
) -> Option<HostCostPolicyV2> {
    // Legacy decoders do not promise bounded scratch/allocation behavior.
    let (decoder, raw_bound, _, _) = sequence.credited_output.as_ref()?.decoder.cost_policy()?;
    Some(HostCostPolicyV2 {
        empirical_content_domain: empirical_content_domain(sequence),
        categorical_signature: policy_signature(sequence, tokenizer, true)?,
        decoder_text_bytes_per_token: u64::try_from(decoder.requirements(1).ok()?.text_bytes)
            .ok()?,
        decoder_scratch_bytes_per_token: u64::try_from(decoder.max_scratch_bytes_per_token())
            .ok()?,
        raw_token_bytes_bound: u64::try_from(raw_bound).ok()?,
    })
}

/// Narrow, installed plain-text algorithm domain. It deliberately permits
/// UTF-8 pending states: their retries and decoder fallback are measured noise
/// in the new whole-wave empirical model, not an asserted constant host cost.
fn empirical_content_domain(sequence: &SequenceState) -> Option<HostContentDomainV1> {
    use crate::continuous_engine::{ResponseCompletionState, SequenceSamplingHistoryScope};
    use ferrum_types::{
        ModelOutputProtocol, ResponseCompletionBoundary, ResponseFormat, StructuredOutputStart,
    };
    let p = &sequence.sampling_params;
    (p.temperature == 0.0
        && p.top_p == 1.0
        && p.top_k.is_none()
        && p.repetition_penalty == 1.0
        && p.presence_penalty == 0.0
        && p.frequency_penalty == 0.0
        && p.min_p.is_none()
        && p.tfs.is_none()
        && p.typical_p.is_none()
        && p.mirostat.is_none()
        && matches!(p.response_format, ResponseFormat::Text)
        && p.model_output_protocol == ModelOutputProtocol::Text
        && matches!(p.structured_output_start, StructuredOutputStart::Immediate)
        && matches!(
            p.response_completion_boundary,
            ResponseCompletionBoundary::Immediate
        )
        && matches!(
            sequence.response_completion_state,
            ResponseCompletionState::Satisfied
        )
        && matches!(
            sequence.sampling_history.scope(),
            SequenceSamplingHistoryScope::FullGeneration
        )
        && sequence.structured_output_processor.is_none()
        // Normal construction keeps model EOS within stop_token_ids. Check
        // both resolved fields so an inconsistent state cannot advertise the
        // empirical domain while stop_reason can still terminate on EOS.
        && sequence.model_eos_token_ids.is_empty()
        && sequence.stop_token_ids.is_empty()
        && sequence.user_stop_token_ids.is_empty()
        && sequence.stop_text_seqs.is_empty()
        && p.stop_sequences.is_empty())
    .then_some(HostContentDomainV1::PlainTextGreedyV1)
}

fn policy_signature(
    sequence: &SequenceState,
    tokenizer: &dyn Tokenizer,
    numeric: bool,
) -> Option<[u8; 32]> {
    policy_signature_with_algorithm(
        sequence,
        tokenizer,
        numeric,
        Some(HOST_OUTPUT_ALGORITHM_REVISION),
    )
}

// Private factorization also lets tests reconstruct the exact pre-revision
// digest (None), without deserializing or granting authority to old receipts.
fn policy_signature_with_algorithm(
    sequence: &SequenceState,
    tokenizer: &dyn Tokenizer,
    numeric: bool,
    algorithm_revision: Option<&str>,
) -> Option<[u8; 32]> {
    let tokenizer_identity = tokenizer.host_output_policy_identity()?;
    let params = &sequence.sampling_params;
    params.validate().ok()?;
    if params.tfs.is_some() || params.typical_p.is_some() || params.mirostat.is_some() {
        return None;
    }
    // The sequence constructor installs these built-in plans. Reject drift
    // from that contract rather than hash stale requested sampling settings.
    let expected = SamplingConfig::from_params(params);
    let actual = &sequence.sampling_plan;
    if actual.sampler.name() != expected.sampler.name()
        || actual.sampler.is_deterministic() != expected.sampler.is_deterministic()
        || actual.processor_chain.processor_names() != expected.processor_chain.processor_names()
    {
        return None;
    }
    if sequence.structured_output_processor.is_some()
        != sequence.original_request.requires_structured_output()
    {
        return None;
    }

    let mut digest = PolicyDigest(Sha256::new());
    digest.0.update(if numeric {
        b"ferrum.engine.host-policy.v2\0"
    } else {
        b"ferrum.engine.host-policy.v1\0"
    });
    if let Some(revision) = algorithm_revision {
        digest.field(&revision)?;
    }
    digest.0.update(tokenizer_identity);
    // Keep every numerical/structured sampling field, while explicitly
    // excluding the stochastic realization from the host execution contract.
    if numeric {
        // Exhaustive destructuring makes a newly added sampling field a compile
        // error until its cost semantics are explicitly classified.
        let ferrum_types::SamplingParams {
            max_tokens: _,
            seed: _,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            stop_sequences,
            min_p,
            tfs,
            typical_p,
            mirostat,
            response_format,
            structured_output_start,
            response_completion_boundary,
            model_output_protocol,
        } = params;
        digest.field(&(
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            stop_sequences,
            min_p,
        ))?;
        digest.field(&(
            tfs,
            typical_p,
            mirostat,
            response_format,
            structured_output_start,
            response_completion_boundary,
            model_output_protocol,
        ))?;
    } else {
        let mut sampling = params.clone();
        sampling.seed = None;
        digest.field(&sampling)?;
    }
    digest.field(&actual.sampler.name())?;
    digest.field(&actual.processor_chain.processor_names())?;
    digest.field(&sequence.structured_output_processor.is_some())?;
    digest.field(&api_policy(sequence.original_request.api_request.as_ref()))?;
    digest.field(&sequence.original_request.evidence_request)?;

    // Hash actual resolved constraints, with deterministic set ordering. These
    // also cover ignore-EOS and initial-mask metadata without copying metadata
    // that could contain prompt text or request-local correlation identifiers.
    digest.set(&sequence.stop_token_ids)?;
    digest.set(&sequence.user_stop_token_ids)?;
    digest.field(&sequence.model_eos_token_ids)?;
    digest.field(&sequence.stop_text_seqs)?;
    digest.set(&sequence.forbidden_token_ids)?;
    digest.set(&sequence.initial_forbidden_token_ids)?;
    digest.set(&sequence.allowed_extended_token_ids)?;
    digest.field(&sequence.tokenizer_base_vocab_size)?;
    digest.field(
        &sequence
            .argmax_token_mask
            .as_ref()
            .map(|mask| mask.valid_token_mask.as_ref()),
    )?;
    digest.field(
        &sequence
            .initial_argmax_token_mask
            .as_ref()
            .map(|mask| mask.valid_token_mask.as_ref()),
    )?;

    if let Some(output) = &sequence.credited_output {
        if sequence.stream_sender.is_some() || sequence.response_sender.is_some() {
            return None;
        }
        let (decoder, raw_bound, max_tokens, codec) = output.decoder.cost_policy()?;
        if tokenizer.bounded_decode_bound() != Some(decoder)
            || tokenizer
                .bounded_token_bytes_bound()
                .map(|bound| bound.get())
                != Some(raw_bound)
            || max_tokens != params.max_tokens
        {
            return None;
        }
        digest.field(&"credited-owner.v1/bounded-decode-wrapper.v1/full-history-projection.v2")?;
        digest.field(&decoder.requirements(1).ok()?.text_bytes)?;
        digest.field(&decoder.max_scratch_bytes_per_token())?;
        digest.field(&raw_bound)?;
        if !numeric {
            digest.field(&max_tokens)?;
        } else {
            // Actual admitted incremental algorithm, not just a tokenizer name.
            digest.field(&output.decoder.cost_incremental_policy())?;
        }
        digest.field(&codec)?;
    } else {
        match (
            sequence.stream_sender.is_some(),
            sequence.response_sender.is_some(),
        ) {
            (true, false) => digest.field(&"legacy-stream.v1/full-history-projection.v2")?,
            (false, true) => digest.field(&"legacy-response.v1/full-history-projection.v2")?,
            _ => return None,
        }
    }
    Some(digest.0.finalize().into())
}

/// Only response/tool contracts enter the key. Chat messages, completion
/// prompt and unrelated request metadata never cross this boundary.
fn api_policy(request: Option<&ApiRequest>) -> serde_json::Value {
    match request {
        None => serde_json::json!({"kind": "none"}),
        Some(ApiRequest::Completion(completion)) => serde_json::json!({
            "kind": "completion",
            "response_format": &completion.response_format,
        }),
        Some(ApiRequest::Chat(chat)) => serde_json::json!({
            "kind": "chat",
            "tools": &chat.tools,
            "tool_choice": &chat.tool_choice,
            "tool_call_protocol": &chat.tool_call_protocol,
            "legacy_functions": &chat.legacy_functions,
            "legacy_function_call": &chat.legacy_function_call,
            "response_format": &chat.response_format,
            "stream_options": &chat.stream_options,
        }),
    }
}

struct PolicyDigest(Sha256);

impl PolicyDigest {
    fn field(&mut self, value: &impl Serialize) -> Option<()> {
        serde_json::to_writer(&mut *self, value).ok()?;
        // Serialized JSON escapes control characters, so this separator
        // cannot collide with a byte embedded in a field's JSON representation.
        self.0.update([0]);
        Some(())
    }

    fn set(&mut self, values: &HashSet<u32>) -> Option<()> {
        let mut ordered: Vec<_> = values.iter().copied().collect();
        ordered.sort_unstable();
        self.field(&ordered)
    }
}

impl io::Write for PolicyDigest {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
