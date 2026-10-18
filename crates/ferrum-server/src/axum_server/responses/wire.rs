//! Borrowed terminal wire views. The Chat/history owner remains live through encoding.
use super::{ResponseContext, ResponseToolName};
use crate::openai::{AssistantMessagePhase, Usage};
use serde::{ser::SerializeMap, Serialize, Serializer};

pub(super) struct ResponseView<'a, T> {
    pub context: &'a ResponseContext,
    pub status: &'a str,
    pub output: T,
    pub usage: Option<&'a Usage>,
    pub completed_at: Option<u64>,
}
impl<T: Serialize> Serialize for ResponseView<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let c = self.context;
        let mut m = serializer.serialize_map(None)?;
        m.serialize_entry("id", &c.id)?;
        m.serialize_entry("object", "response")?;
        m.serialize_entry("created_at", &c.created_at)?;
        m.serialize_entry("completed_at", &self.completed_at)?;
        m.serialize_entry("status", self.status)?;
        m.serialize_entry("background", &false)?;
        m.serialize_entry("error", &Option::<()>::None)?;
        m.serialize_entry(
            "incomplete_details",
            &if self.status == "incomplete" {
                Some(serde_json::json!({"reason":"max_output_tokens"}))
            } else {
                None
            },
        )?;
        m.serialize_entry("instructions", &c.instructions)?;
        m.serialize_entry("max_output_tokens", &c.max_output_tokens)?;
        m.serialize_entry("max_tool_calls", &Option::<()>::None)?;
        m.serialize_entry("metadata", &c.metadata)?;
        m.serialize_entry("model", &c.model)?;
        m.serialize_entry("output", &self.output)?;
        m.serialize_entry("parallel_tool_calls", &c.parallel_tool_calls)?;
        m.serialize_entry("previous_response_id", &Option::<()>::None)?;
        m.serialize_entry("reasoning", &c.reasoning)?;
        m.serialize_entry("service_tier", "default")?;
        m.serialize_entry("store", &false)?;
        m.serialize_entry("temperature", &c.temperature)?;
        m.serialize_entry("text", &c.text)?;
        m.serialize_entry("tool_choice", &c.tool_choice)?;
        m.serialize_entry("tools", &c.tools)?;
        m.serialize_entry("top_logprobs", &0)?;
        m.serialize_entry("top_p", &c.top_p)?;
        m.serialize_entry("truncation", "disabled")?;
        m.serialize_entry("usage", &self.usage.map(super::response_usage))?;
        m.serialize_entry("user", &c.user)?;
        m.serialize_entry("presence_penalty", &c.presence_penalty)?;
        m.serialize_entry("frequency_penalty", &c.frequency_penalty)?;
        m.serialize_entry("prompt_cache_key", &c.prompt_cache_key)?;
        m.serialize_entry("safety_identifier", &Option::<()>::None)?;
        m.end()
    }
}

pub(super) enum Item<'a> {
    Reasoning {
        id: &'a str,
        text: &'a str,
        status: &'a str,
        encrypted: bool,
    },
    Message {
        id: &'a str,
        text: &'a str,
        status: &'a str,
        phase: AssistantMessagePhase,
    },
    Function {
        id: &'a str,
        call_id: &'a str,
        name: &'a ResponseToolName,
        arguments: &'a str,
        status: &'a str,
    },
}

#[derive(Serialize)]
pub(super) struct TextPart<'a> {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: &'a str,
    pub annotations: [(); 0],
    pub logprobs: [(); 0],
}
#[derive(Serialize)]
pub(super) struct ReasoningPart<'a> {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: &'a str,
}
impl Serialize for Item<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(None)?;
        match self {
            Self::Reasoning {
                id,
                text,
                status,
                encrypted,
            } => {
                m.serialize_entry("id", id)?;
                m.serialize_entry("type", "reasoning")?;
                m.serialize_entry("status", status)?;
                m.serialize_entry("summary", &[(); 0])?;
                m.serialize_entry(
                    "content",
                    &[ReasoningPart {
                        kind: "reasoning_text",
                        text,
                    }],
                )?;
                if *encrypted {
                    m.serialize_entry("encrypted_content", &Option::<()>::None)?;
                }
            }
            Self::Message {
                id,
                text,
                status,
                phase,
            } => {
                m.serialize_entry("id", id)?;
                m.serialize_entry("type", "message")?;
                m.serialize_entry("status", status)?;
                m.serialize_entry("role", "assistant")?;
                m.serialize_entry("phase", phase)?;
                m.serialize_entry(
                    "content",
                    &[TextPart {
                        kind: "output_text",
                        text,
                        annotations: [],
                        logprobs: [],
                    }],
                )?;
            }
            Self::Function {
                id,
                call_id,
                name,
                arguments,
                status,
            } => {
                m.serialize_entry("id", id)?;
                m.serialize_entry("type", "function_call")?;
                m.serialize_entry("status", status)?;
                m.serialize_entry("call_id", call_id)?;
                m.serialize_entry("name", &name.name)?;
                m.serialize_entry("arguments", arguments)?;
                if let Some(namespace) = &name.namespace {
                    m.serialize_entry("namespace", namespace)?;
                }
            }
        }
        m.end()
    }
}

#[derive(Serialize)]
pub(super) struct Event<T> {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub sequence_number: u64,
    #[serde(flatten)]
    pub fields: T,
}

#[derive(Serialize)]
pub(super) struct ItemEvent<T> {
    pub output_index: usize,
    pub item: T,
}
#[derive(Serialize)]
pub(super) struct TextEvent<'a> {
    pub item_id: &'a str,
    pub output_index: usize,
    pub content_index: usize,
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<[(); 0]>,
}
#[derive(Serialize)]
pub(super) struct PartEvent<'a, T> {
    pub item_id: &'a str,
    pub output_index: usize,
    pub content_index: usize,
    pub part: T,
}
#[derive(Serialize)]
pub(super) struct ResponseEvent<T> {
    pub response: T,
}
#[derive(Serialize)]
pub(super) struct ArgumentsEvent<'a> {
    pub item_id: &'a str,
    pub output_index: usize,
    pub call_id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<&'a str>,
}
#[derive(Serialize)]
pub(super) struct ArgumentsDelta<'a> {
    pub item_id: &'a str,
    pub output_index: usize,
    pub delta: &'a str,
}
