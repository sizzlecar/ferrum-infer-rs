//! Acceptance capacity derived from the actual waiting queue under its write
//! lock. No second ledger can outlive a cancellation, admission or queue move.
use super::*;
use ferrum_types::{SloMode, SloTimeAdmissionPolicy};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct WaitingPromptUsage {
    requests: usize,
    tokens: usize,
    /// Rendered prompt UTF-8 payload, not allocation capacity or total RSS.
    bytes: usize,
}

impl WaitingPromptUsage {
    fn from_tokenized_request(request: &InferenceRequest) -> Result<Self> {
        // All engine ingress paths overwrite this field AFTER actual model
        // tokenization and BEFORE scheduler submission. Direct scheduler users
        // must provide the same internal contract; estimates/missing values do
        // not acquire waiting capacity. Never use prompt_token_estimate here.
        let tokens = request
            .metadata
            .get(PROMPT_TOKENS_METADATA_KEY)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                FerrumError::invalid_request(
                    "SLO waiting admission requires the engine-tokenized prompt count",
                )
            })?;
        Ok(Self {
            requests: 1,
            tokens,
            bytes: request.prompt.len(),
        })
    }

    fn checked_add(self, other: Self) -> Result<Self> {
        let overflow = || FerrumError::resource_exhausted("SLO waiting prompt accounting overflow");
        Ok(Self {
            requests: self
                .requests
                .checked_add(other.requests)
                .ok_or_else(overflow)?,
            tokens: self.tokens.checked_add(other.tokens).ok_or_else(overflow)?,
            bytes: self.bytes.checked_add(other.bytes).ok_or_else(overflow)?,
        })
    }
}

pub(super) fn check(
    config: &SchedulerConfig,
    waiting: &DynamicAdmissionQueue<ContinuousBatchRequest>,
    incoming: &InferenceRequest,
) -> Result<()> {
    let enabled = config.slo.mode == SloMode::Enforce
        && config.slo.admission.time_policy == SloTimeAdmissionPolicy::CompleteRequests;
    if waiting.len() >= config.max_waiting_requests {
        return Err(if enabled {
            FerrumError::resource_exhausted("scheduler waiting request capacity exhausted")
        } else {
            FerrumError::scheduler("Queue is full, cannot accept more requests")
        });
    }
    if !enabled {
        return Ok(());
    }
    let mut usage = WaitingPromptUsage::from_tokenized_request(incoming)?;
    for queued in waiting.iter() {
        usage = usage.checked_add(WaitingPromptUsage::from_tokenized_request(
            &queued.inner.request,
        )?)?;
    }
    let policy = &config.slo.admission;
    for (name, proposed, limit) in [
        (
            "requests",
            usage.requests,
            policy.max_waiting_requests.get(),
        ),
        (
            "prompt tokens",
            usage.tokens,
            policy.max_waiting_prompt_tokens.get(),
        ),
        (
            "prompt UTF-8 bytes",
            usage.bytes,
            policy.max_waiting_prompt_bytes.get(),
        ),
    ] {
        if proposed > limit {
            return Err(FerrumError::resource_exhausted(format!(
                "SLO waiting {name} capacity exhausted: proposed {proposed}, limit {limit}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
