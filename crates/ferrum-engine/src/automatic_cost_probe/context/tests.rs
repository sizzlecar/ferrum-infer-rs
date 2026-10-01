use super::*;
use ferrum_types::{SpecialTokens, TokenId};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct Words {
    specials: SpecialTokens,
    stride: usize,
}
impl Tokenizer for Words {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        assert!(add_special);
        Ok(vec![
            TokenId::new(1);
            text.split_whitespace().count() * self.stride.max(1)
        ])
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
        &self.specials
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
#[derive(Debug)]
struct Renderer {
    calls: Arc<AtomicUsize>,
    change_policy: bool,
}
impl AutomaticCostProbePromptRenderer for Renderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut request = base().resolved_request()?;
        request.prompt = format!("user:{text} assistant:");
        if self.change_policy {
            request.sampling_params.temperature = 0.5;
        }
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText)
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        Some(std::mem::size_of::<Self>())
    }
}
fn base() -> AutomaticCostProbeTemplate {
    let mut request = ferrum_types::InferenceRequest::new("user: original assistant:", "fixture");
    request.stream = true;
    AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()
}
fn budget() -> AutomaticCostProbePromptBudget {
    AutomaticCostProbePromptBudget::new(AutomaticCostProbePromptLimits {
        maximum_attempts: NonZeroUsize::new(64).unwrap(),
        maximum_rendered_bytes: NonZeroUsize::new(65_536).unwrap(),
        maximum_tokenized_tokens: NonZeroUsize::new(16_384).unwrap(),
        maximum_retained_bytes: NonZeroUsize::new(65_536).unwrap(),
    })
}
#[test]
fn automatic_probe_prompt_window_uses_checked_tokens_and_one_shared_budget() {
    let calls = Arc::new(AtomicUsize::new(0));
    let original = base().with_prompt_renderer(Arc::new(Renderer {
        calls: calls.clone(),
        change_policy: false,
    }));
    let original_bytes = original.serialized_request().to_vec();
    let mut budget = budget();
    for target in [63, 64] {
        let variant = original
            .for_prompt_token_window(
                &Words::default(),
                NonZeroUsize::new(target).unwrap(),
                NonZeroUsize::new(target).unwrap(),
                &mut budget,
            )
            .unwrap()
            .unwrap();
        assert_eq!(variant.prompt_tokens.get(), target);
        assert!(variant.template.prompt().starts_with("user:"));
        assert!(variant.template.prompt().ends_with("assistant:"));
        assert!(variant.template.prompt_renderer.is_none());
    }
    assert_eq!(budget.usage().attempts, calls.load(Ordering::Relaxed));
    assert!(budget.usage().tokenized_tokens >= 127);
    assert!(budget.usage().retained_bytes > 0);
    assert_eq!(original.serialized_request(), original_bytes);
}
#[test]
fn automatic_probe_prompt_unreachable_token_window_stays_uncovered() {
    let original = base().with_prompt_renderer(Arc::new(Renderer {
        calls: Default::default(),
        change_policy: false,
    }));
    let target = NonZeroUsize::new(63).unwrap();
    let missing = original
        .for_prompt_token_window(
            &Words {
                stride: 2,
                ..Default::default()
            },
            target,
            target,
            &mut budget(),
        )
        .unwrap();
    assert!(
        missing.is_none(),
        "even token lengths cannot be truncated into an odd frontier"
    );
    assert!(base()
        .for_prompt_token_window(&Words::default(), target, target, &mut budget())
        .unwrap()
        .is_none());
}
#[test]
fn automatic_probe_prompt_rejects_policy_changes_and_shared_budget_exhaustion() {
    let target = NonZeroUsize::new(64).unwrap();
    let changed = base().with_prompt_renderer(Arc::new(Renderer {
        calls: Default::default(),
        change_policy: true,
    }));
    assert!(changed
        .for_prompt_token_window(&Words::default(), target, target, &mut budget())
        .is_err());
    let calls = Arc::new(AtomicUsize::new(0));
    let original = base().with_prompt_renderer(Arc::new(Renderer {
        calls: calls.clone(),
        change_policy: false,
    }));
    let mut budget = budget();
    budget.limits.maximum_attempts = NonZeroUsize::MIN;
    assert!(original
        .for_prompt_token_window(&Words::default(), target, target, &mut budget)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(original
        .for_prompt_token_window(&Words::default(), target, target, &mut budget)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn automatic_probe_prompt_post_work_exhaustion_cannot_reset_shared_budget() {
    let target = NonZeroUsize::new(64).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let original = base().with_prompt_renderer(Arc::new(Renderer {
        calls: calls.clone(),
        change_policy: false,
    }));
    let mut allowance = budget();
    allowance.limits.maximum_tokenized_tokens = NonZeroUsize::MIN;
    assert!(original
        .for_prompt_token_window(&Words::default(), target, target, &mut allowance)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(allowance.usage().tokenized_tokens > 1);
    assert!(original
        .for_prompt_token_window(&Words::default(), target, target, &mut allowance)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[derive(Debug, Clone, Copy)]
enum ChatFault {
    None,
    UserMirror,
    TokenMask,
}
#[derive(Debug)]
struct ChatRenderer(ChatFault);
impl AutomaticCostProbePromptRenderer for ChatRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        let mut request = chat_request(text);
        match self.0 {
            ChatFault::None => {}
            ChatFault::UserMirror => {
                request.metadata.get_mut("openai_messages").unwrap()[0]["content"] =
                    serde_json::json!("different content");
            }
            ChatFault::TokenMask => {
                request.metadata.insert(
                    "ferrum_initial_forbidden_token_texts".into(),
                    serde_json::json!(["changed"]),
                );
            }
        }
        AutomaticCostProbeTemplate::new(
            request,
            AutomaticCostProbeOutput::ApiChatSse {
                include_usage: false,
            },
        )
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        Some(std::mem::size_of::<Self>())
    }
}
fn chat_request(text: &str) -> InferenceRequest {
    use ferrum_types::{ApiChatMessage, ApiChatRequest, ApiMessageRole};
    let message = ApiChatMessage {
        role: ApiMessageRole::User,
        content: text.into(),
        name: None,
        tool_calls: vec![],
        tool_call_id: None,
        function_call: None,
    };
    let mut request = InferenceRequest::new(format!("user: {text} assistant:"), "fixture");
    request.stream = true;
    request
        .metadata
        .insert("openai_messages".into(), serde_json::json!([message]));
    request.metadata.insert(
        ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
        serde_json::json!(false),
    );
    request.metadata.insert(
        "ferrum_initial_forbidden_token_texts".into(),
        serde_json::json!([]),
    );
    request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: vec![message],
        tools: vec![],
        tool_choice: None,
        tool_call_protocol: Default::default(),
        legacy_functions: vec![],
        legacy_function_call: None,
        response_format: None,
        stream_options: None,
    }));
    request
}

#[test]
fn automatic_probe_prompt_chat_mirror_changes_only_checked_user_content() {
    let target = NonZeroUsize::new(64).unwrap();
    for fault in [ChatFault::None, ChatFault::UserMirror, ChatFault::TokenMask] {
        let original = AutomaticCostProbeTemplate::new(
            chat_request("original"),
            AutomaticCostProbeOutput::ApiChatSse {
                include_usage: false,
            },
        )
        .unwrap()
        .with_prompt_renderer(Arc::new(ChatRenderer(fault)));
        let result =
            original.for_prompt_token_window(&Words::default(), target, target, &mut budget());
        match fault {
            ChatFault::None => assert_eq!(result.unwrap().unwrap().prompt_tokens, target),
            ChatFault::UserMirror => assert!(result
                .unwrap_err()
                .to_string()
                .contains("metadata.openai_messages")),
            ChatFault::TokenMask => assert!(result
                .unwrap_err()
                .to_string()
                .contains("ferrum_initial_forbidden_token_texts")),
        }
    }
}
