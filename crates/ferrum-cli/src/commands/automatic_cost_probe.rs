//! Shared retained representation for product-owned cold prompt renderers.
//! Serialize only the real model template and its resolved capabilities, never
//! an engine, registry, tokenizer, runtime or full EngineConfig.
use ferrum_server::chat_template::ModelChatTemplate;
use ferrum_types::{
    ApiToolCallProtocol, FerrumError, ModelOutputProtocol, ModelReasoningProtocol, ReasoningEffort,
    ReasoningEffortSupport, Result,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) struct PromptModel(Arc<[u8]>);

#[derive(serde::Serialize, serde::Deserialize)]
struct Model {
    template: String,
    source: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    tool_call_protocol: ApiToolCallProtocol,
    output_protocol: ModelOutputProtocol,
    reasoning_protocol: ModelReasoningProtocol,
    reasoning_default_enabled: bool,
    reasoning_efforts: Option<Vec<ReasoningEffort>>,
}
impl PromptModel {
    pub(super) fn new(value: &ModelChatTemplate) -> Result<Self> {
        let model = Model {
            template: value.template.clone(),
            source: value.source.clone(),
            bos_token: value.bos_token.clone(),
            eos_token: value.eos_token.clone(),
            tool_call_protocol: value.tool_call_protocol,
            output_protocol: value.output_protocol,
            reasoning_protocol: value.reasoning_protocol,
            reasoning_default_enabled: value.reasoning_default_enabled,
            reasoning_efforts: value
                .reasoning_effort_support
                .declared_efforts()
                .map(|values| values.iter().copied().collect()),
        };
        Ok(Self(
            serde_json::to_vec(&model)
                .map_err(|e| FerrumError::internal(format!("probe model renderer encoding: {e}")))?
                .into(),
        ))
    }
    pub(super) fn resolve(&self) -> Result<ModelChatTemplate> {
        let model: Model = serde_json::from_slice(&self.0)
            .map_err(|e| FerrumError::internal(format!("probe model renderer decoding: {e}")))?;
        Ok(ModelChatTemplate {
            template: model.template,
            source: model.source,
            bos_token: model.bos_token,
            eos_token: model.eos_token,
            tool_call_protocol: model.tool_call_protocol,
            output_protocol: model.output_protocol,
            reasoning_protocol: model.reasoning_protocol,
            reasoning_default_enabled: model.reasoning_default_enabled,
            reasoning_effort_support: model
                .reasoning_efforts
                .map_or(ReasoningEffortSupport::Unknown, |values| {
                    ReasoningEffortSupport::Declared(values.into_iter().collect())
                }),
        })
    }
    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.0.len())?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use ferrum_interfaces::Tokenizer;
    use ferrum_types::{SpecialTokens, TokenId};
    #[derive(Default)]
    pub(crate) struct Words(SpecialTokens);
    impl Tokenizer for Words {
        fn encode(&self, text: &str, special: bool) -> Result<Vec<TokenId>> {
            assert!(special);
            Ok(vec![TokenId::new(1); text.split_whitespace().count()])
        }
        fn decode(&self, _: &[TokenId], _: bool) -> Result<String> {
            unreachable!()
        }
        fn decode_incremental(&self, _: &[TokenId], _: TokenId) -> Result<String> {
            unreachable!()
        }
        fn vocab_size(&self) -> usize {
            2
        }
        fn special_tokens(&self) -> &SpecialTokens {
            &self.0
        }
        fn token_id(&self, _: &str) -> Option<TokenId> {
            None
        }
        fn token_text(&self, _: TokenId) -> Option<&str> {
            None
        }
        fn info(&self) -> ferrum_interfaces::tokenizer::TokenizerInfo {
            unreachable!("prompt geometry only uses encode")
        }
    }
    pub(crate) fn variant(
        original: &ferrum_engine::AutomaticCostProbeTemplate,
    ) -> ferrum_engine::AutomaticCostProbePromptVariant {
        use ferrum_engine::{AutomaticCostProbePromptBudget, AutomaticCostProbePromptLimits};
        use std::num::NonZeroUsize;
        let mut budget = AutomaticCostProbePromptBudget::new(AutomaticCostProbePromptLimits {
            maximum_attempts: NonZeroUsize::new(64).unwrap(),
            maximum_rendered_bytes: NonZeroUsize::new(65_536).unwrap(),
            maximum_tokenized_tokens: NonZeroUsize::new(16_384).unwrap(),
            maximum_retained_bytes: NonZeroUsize::new(65_536).unwrap(),
        });
        let target = NonZeroUsize::new(128).unwrap();
        let variant = original
            .for_prompt_token_window(&Words::default(), target, target, &mut budget)
            .unwrap()
            .unwrap();
        assert_eq!(variant.prompt_tokens, target);
        variant
    }
}
