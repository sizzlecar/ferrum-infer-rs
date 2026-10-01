//! Cold tokenizer-checked prompt geometry. Only the product renderer owns chat
//! framing; this module supplies ordinary user text under one shared allowance.
use super::*;
use ferrum_interfaces::Tokenizer;

pub trait AutomaticCostProbePromptRenderer: std::fmt::Debug + Send + Sync {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate>;
    fn retained_payload_bytes(&self) -> Option<usize>;
}

#[derive(Clone, Copy, Debug)]
pub struct AutomaticCostProbePromptLimits {
    pub maximum_attempts: NonZeroUsize,
    pub maximum_rendered_bytes: NonZeroUsize,
    pub maximum_tokenized_tokens: NonZeroUsize,
    pub maximum_retained_bytes: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutomaticCostProbePromptUsage {
    pub attempts: usize,
    /// Cumulative user-text and rendered-prompt bytes, including failed tries.
    pub rendered_bytes: usize,
    pub tokenized_tokens: usize,
    pub retained_bytes: usize,
}

#[derive(Debug)]
pub struct AutomaticCostProbePromptBudget {
    limits: AutomaticCostProbePromptLimits,
    usage: AutomaticCostProbePromptUsage,
    exhausted: bool,
}
enum Charge {
    Attempt,
    Rendered,
    Tokenized,
    Retained,
}
impl AutomaticCostProbePromptBudget {
    pub fn new(limits: AutomaticCostProbePromptLimits) -> Self {
        Self {
            limits,
            usage: Default::default(),
            exhausted: false,
        }
    }
    pub fn usage(&self) -> AutomaticCostProbePromptUsage {
        self.usage
    }
    fn charge(&mut self, kind: Charge, added: usize) -> Result<()> {
        let (value, limit) = match kind {
            Charge::Attempt => (&mut self.usage.attempts, self.limits.maximum_attempts),
            Charge::Rendered => (
                &mut self.usage.rendered_bytes,
                self.limits.maximum_rendered_bytes,
            ),
            Charge::Tokenized => (
                &mut self.usage.tokenized_tokens,
                self.limits.maximum_tokenized_tokens,
            ),
            Charge::Retained => (
                &mut self.usage.retained_bytes,
                self.limits.maximum_retained_bytes,
            ),
        };
        let total = value.checked_add(added);
        *value = total.unwrap_or(usize::MAX);
        self.exhausted |= total.is_none_or(|n| n > limit.get());
        if self.exhausted {
            return Err(FerrumError::resource_exhausted(
                "automatic probe prompt search allowance exhausted",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AutomaticCostProbePromptVariant {
    pub template: AutomaticCostProbeTemplate,
    pub prompt_tokens: NonZeroUsize,
}

impl AutomaticCostProbeTemplate {
    /// Find a product-rendered prompt inside the inclusive tokenizer window.
    /// The caller derives this window from real provider geometry, effective
    /// context and its declared prefix/suffix. No token slicing or estimated
    /// length can turn an unsuccessful search into covered geometry.
    pub fn for_prompt_token_window(
        &self,
        tokenizer: &dyn Tokenizer,
        minimum: NonZeroUsize,
        maximum: NonZeroUsize,
        budget: &mut AutomaticCostProbePromptBudget,
    ) -> Result<Option<AutomaticCostProbePromptVariant>> {
        if minimum > maximum {
            return Err(FerrumError::invalid_request(
                "probe prompt token window is reversed",
            ));
        }
        if budget.exhausted {
            return Err(FerrumError::resource_exhausted(
                "automatic probe prompt search allowance exhausted",
            ));
        }
        let Some(renderer) = &self.prompt_renderer else {
            return Ok(None);
        };
        let original = self.resolved_request()?;
        let original_sampling = serde_json::to_value(&original.sampling_params)
            .map_err(|e| FerrumError::internal(format!("probe sampling identity: {e}")))?;
        let original_metadata = normalized_metadata(&original)?;
        const UNIT: &str = " object";
        let remaining = budget
            .limits
            .maximum_rendered_bytes
            .get()
            .saturating_sub(budget.usage.rendered_bytes)
            .min(
                budget
                    .limits
                    .maximum_retained_bytes
                    .get()
                    .saturating_sub(budget.usage.retained_bytes),
            );
        let maximum_units = remaining / UNIT.len();
        if maximum_units == 0 {
            return Err(FerrumError::resource_exhausted(
                "probe prompt search has no text capacity",
            ));
        }
        let mut lower = 1usize;
        let mut upper = maximum_units;
        let mut units = 1usize;
        let mut bracketed = false;
        while lower <= upper {
            budget.charge(Charge::Attempt, 1)?;
            budget.charge(Charge::Rendered, units * UNIT.len())?;
            let text = UNIT.repeat(units);
            let mut candidate = renderer.render_user_text(&text)?;
            candidate.prompt_renderer = None;
            budget.charge(Charge::Rendered, candidate.prompt().len())?;
            let retained = candidate.retained_payload_bytes().ok_or_else(|| {
                FerrumError::resource_exhausted("probe prompt retained size overflow")
            })?;
            if retained
                .checked_add(budget.usage.retained_bytes)
                .is_none_or(|n| n > budget.limits.maximum_retained_bytes.get())
            {
                budget.exhausted = true;
                return Err(FerrumError::resource_exhausted(
                    "probe prompt retained capacity exceeded",
                ));
            }
            let request = candidate.resolved_request()?;
            let candidate_sampling = serde_json::to_value(&request.sampling_params)
                .map_err(|e| FerrumError::internal(format!("probe sampling identity: {e}")))?;
            let candidate_metadata = normalized_metadata(&request)?;
            let changed = if candidate.output != self.output {
                Some("output".to_owned())
            } else if candidate.response_model != self.response_model {
                Some("response_model".to_owned())
            } else if request.model_id != original.model_id {
                Some("model_id".to_owned())
            } else if request.stream != original.stream {
                Some("stream".to_owned())
            } else if candidate_metadata != original_metadata {
                let key = candidate_metadata
                    .keys()
                    .chain(original_metadata.keys())
                    .filter(|key| candidate_metadata.get(*key) != original_metadata.get(*key))
                    .min()
                    .expect("unequal metadata has a changed key");
                Some(format!("metadata.{key}"))
            } else if candidate_sampling != original_sampling {
                Some("sampling_params".to_owned())
            } else if normalized_api(request.api_request)
                != normalized_api(original.api_request.clone())
            {
                Some("api_request".to_owned())
            } else {
                None
            };
            if let Some(field) = changed {
                return Err(FerrumError::invalid_request(format!(
                    "probe prompt renderer changed the original product policy: {field}"
                )));
            }
            let tokens = tokenizer.encode(candidate.prompt(), true)?.len();
            budget.charge(Charge::Tokenized, tokens)?;
            if (minimum.get()..=maximum.get()).contains(&tokens) {
                budget.charge(Charge::Retained, retained)?;
                return Ok(Some(AutomaticCostProbePromptVariant {
                    template: candidate,
                    prompt_tokens: NonZeroUsize::new(tokens).unwrap(),
                }));
            }
            if tokens < minimum.get() {
                lower = units.saturating_add(1);
                if lower > upper {
                    break;
                }
                units = if bracketed {
                    lower + (upper - lower) / 2
                } else {
                    units.saturating_mul(2).min(upper)
                };
            } else {
                if units == 1 {
                    break;
                }
                upper = units - 1;
                bracketed = true;
                if lower > upper {
                    break;
                }
                units = lower + (upper - lower) / 2;
            }
        }
        // Arbitrary tokenizers need not be monotonic in repeated user text.
        // Search exhaustion is a reported gap, never a completeness claim.
        Ok(None)
    }
}

fn normalized_metadata(
    request: &InferenceRequest,
) -> Result<std::collections::HashMap<String, serde_json::Value>> {
    let mut metadata = request.metadata.clone();
    // The real Chat converter retains the input messages twice. Only checked
    // user-content mirrors may vary with geometry; roles, reasoning, tools,
    // masks and every other metadata value remain part of policy identity.
    if let (Some(ApiRequest::Chat(chat)), Some(messages)) =
        (&request.api_request, metadata.get_mut("openai_messages"))
    {
        let mismatch = || {
            FerrumError::invalid_request(
                "probe metadata.openai_messages does not mirror the actual Chat user content",
            )
        };
        let messages = messages
            .as_array_mut()
            .filter(|m| m.len() == chat.messages.len())
            .ok_or_else(mismatch)?;
        for (message, actual) in messages.iter_mut().zip(&chat.messages) {
            if actual.role != ferrum_types::ApiMessageRole::User {
                continue;
            }
            if message.get("role").and_then(serde_json::Value::as_str) != Some("user")
                || message.get("content").and_then(serde_json::Value::as_str)
                    != Some(actual.content.as_str())
            {
                return Err(mismatch());
            }
            message["content"] = serde_json::Value::String(String::new());
        }
    }
    Ok(metadata)
}

fn normalized_api(mut api: Option<ApiRequest>) -> Option<ApiRequest> {
    match &mut api {
        Some(ApiRequest::Chat(chat)) => {
            for message in &mut chat.messages {
                if message.role == ferrum_types::ApiMessageRole::User {
                    message.content.clear();
                }
            }
        }
        Some(ApiRequest::Completion(completion)) => completion.prompt.clear(),
        None => {}
    }
    api
}

#[cfg(test)]
mod tests;
