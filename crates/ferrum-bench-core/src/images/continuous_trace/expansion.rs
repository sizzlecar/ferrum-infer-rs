//! Independent diagnostic metadata. These records never establish a step,
//! queue interval, native retirement, throughput or image-quality qualification.
use super::*;
use ferrum_types::FinishReason;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageTraceExpansionDiagnosticPolicy {
    Summary,
    CaptureEffectivePrompt { maximum_utf8_bytes: usize },
}
impl<'de> Deserialize<'de> for ImageTraceExpansionDiagnosticPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Empty {}
        #[derive(Deserialize)]
        #[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Summary(Empty),
            CaptureEffectivePrompt { maximum_utf8_bytes: usize },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Summary(_) => Self::Summary,
            Wire::CaptureEffectivePrompt { maximum_utf8_bytes } => {
                Self::CaptureEffectivePrompt { maximum_utf8_bytes }
            }
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceExpansionTokenHistory {
    pub output_tokens_including_eos: u64,
    pub last_token_id: Option<u32>,
    pub protocol_eos_observed: bool,
    pub finish: FinishReason,
}
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageTraceExpansionPromptCapture {
    Disabled,
    Captured {
        text: String,
    },
    CapacityRefused {
        actual_utf8_bytes: usize,
        maximum_utf8_bytes: usize,
    },
    Unavailable,
}
impl<'de> Deserialize<'de> for ImageTraceExpansionPromptCapture {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Empty {}
        #[derive(Deserialize)]
        #[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Disabled(Empty),
            Unavailable(Empty),
            Captured {
                text: String,
            },
            CapacityRefused {
                actual_utf8_bytes: usize,
                maximum_utf8_bytes: usize,
            },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Disabled(_) => Self::Disabled,
            Wire::Unavailable(_) => Self::Unavailable,
            Wire::Captured { text } => Self::Captured { text },
            Wire::CapacityRefused {
                actual_utf8_bytes,
                maximum_utf8_bytes,
            } => Self::CapacityRefused {
                actual_utf8_bytes,
                maximum_utf8_bytes,
            },
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceVisionBindingReceipt {
    pub reused: bool,
    pub producer_job: u64,
    pub producer_request_id: Option<uuid::Uuid>,
    pub output_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTracePromptExpansionObservation {
    pub job: u64,
    pub request_id: Option<uuid::Uuid>,
    pub policy: ImageTraceExpansionDiagnosticPolicy,
    pub prompt_tokens: u64,
    pub protocol_eos_token_id: u32,
    pub initialization_wall_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_owner: Option<ImageTraceSharedOwner>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_binding: Option<ImageTraceVisionBindingReceipt>,
    pub retirement_wall_us: Option<u64>,
    pub executor_retired: bool,
    pub outcome: String,
    pub token_history: Option<ImageTraceExpansionTokenHistory>,
    pub executor_metrics: Option<Value>,
    pub effective_prompt_utf8_bytes: Option<usize>,
    pub effective_prompt_sha256: Option<String>,
    pub capture: ImageTraceExpansionPromptCapture,
}
impl ImageTracePromptExpansionObservation {
    pub(super) fn validate(&self, line: u64) -> Result<(), ImageContinuousTraceError> {
        let invalid = || {
            error(
                line,
                ImageContinuousTraceErrorKind::Invalid,
                "invalid prompt expansion diagnostic",
            )
        };
        let capture_limit = match self.policy {
            ImageTraceExpansionDiagnosticPolicy::Summary => None,
            ImageTraceExpansionDiagnosticPolicy::CaptureEffectivePrompt { maximum_utf8_bytes }
                if maximum_utf8_bytes > 0 && maximum_utf8_bytes <= 4 * 1024 * 1024 =>
            {
                Some(maximum_utf8_bytes)
            }
            _ => return Err(invalid()),
        };
        if self.prompt_tokens == 0
            || !matches!(
                self.outcome.as_str(),
                "expanded" | "failed" | "cancelled" | "retained"
            )
            || (self.outcome == "retained") == self.executor_retired
        {
            return Err(invalid());
        }
        if let Some(binding) = &self.vision_binding {
            // A miss names this request's completed producer. A reused output
            // can outlive a cancelled producer, so no request-link or terminal
            // event is required for the other identity.
            if binding.output_bytes == 0
                || (!binding.reused
                    && (binding.producer_job != self.job
                        || binding.producer_request_id != self.request_id))
            {
                return Err(invalid());
            }
        }
        if let Some(history) = &self.token_history {
            if (history.output_tokens_including_eos == 0) != history.last_token_id.is_none()
                || history.protocol_eos_observed
                    != (history.last_token_id == Some(self.protocol_eos_token_id))
                || (history.finish == FinishReason::EOS && !history.protocol_eos_observed)
            {
                return Err(invalid());
            }
        }
        let expanded = self.outcome == "expanded";
        if expanded
            && self
                .token_history
                .as_ref()
                .is_none_or(|h| h.finish != FinishReason::EOS || !h.protocol_eos_observed)
        {
            return Err(invalid());
        }
        if expanded != self.effective_prompt_utf8_bytes.is_some()
            || expanded != self.effective_prompt_sha256.is_some()
            || self
                .effective_prompt_sha256
                .as_ref()
                .is_some_and(|h| h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()))
            || self
                .executor_metrics
                .as_ref()
                .is_some_and(|m| !m.is_object())
        {
            return Err(invalid());
        }
        match &self.capture {
            ImageTraceExpansionPromptCapture::Disabled if capture_limit.is_none() => (),
            ImageTraceExpansionPromptCapture::Unavailable
                if capture_limit.is_some() && !expanded =>
            {
                ()
            }
            ImageTraceExpansionPromptCapture::CapacityRefused {
                actual_utf8_bytes,
                maximum_utf8_bytes,
            } if expanded
                && capture_limit == Some(*maximum_utf8_bytes)
                && *actual_utf8_bytes > *maximum_utf8_bytes
                && self.effective_prompt_utf8_bytes == Some(*actual_utf8_bytes) =>
            {
                ()
            }
            ImageTraceExpansionPromptCapture::Captured { text }
                if expanded
                    && capture_limit.is_some_and(|limit| text.len() <= limit)
                    && self.effective_prompt_utf8_bytes == Some(text.len())
                    && self.effective_prompt_sha256.as_deref()
                        == Some(crate::images::sha256(text.as_bytes()).as_str()) =>
            {
                ()
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }
}
