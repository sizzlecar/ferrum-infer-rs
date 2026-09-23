//! Immutable finite protocol-marker table, compiled during tokenizer startup.
use super::*;
use ferrum_types::{model_reasoning_markers, ApiToolCallProtocol};

struct Entry {
    text: &'static str,
    tokens: Option<Box<[TokenId]>>,
}
pub(super) struct PreparedCompletionTable {
    entries: Box<[Entry]>,
}
impl PreparedCompletionTable {
    pub(super) fn new(tokenizer: &HfTokenizer) -> Self {
        let started = std::time::Instant::now();
        let mut markers = Vec::new();
        for protocol in [
            ModelOutputProtocol::Text,
            ModelOutputProtocol::GemmaThought,
            ModelOutputProtocol::HarmonyGptOss,
        ] {
            if let Some((open, close)) = model_reasoning_markers(protocol) {
                markers.extend([open, close]);
            }
        }
        for protocol in [
            ApiToolCallProtocol::Json,
            ApiToolCallProtocol::NativeJson,
            ApiToolCallProtocol::FunctionParameterXml,
        ] {
            markers.extend_from_slice(protocol.generated_control_token_texts());
        }
        markers.sort_unstable();
        markers.dedup();
        let entries: Box<[_]> = markers
            .into_iter()
            .map(|text| {
                // Exactly the engine's old token_id-first / encode(false) rule.
                // General HF encode allocates only here in the cold domain. Never
                // introduce a lazy request-time miss path or unbounded marker map.
                let tokens = if let Some(token) = tokenizer.token_to_id(text) {
                    Some(vec![TokenId::new(token)].into_boxed_slice())
                } else {
                    tokenizer.encode(text, false).ok().and_then(|encoding| {
                        (!encoding.get_ids().is_empty()).then(|| {
                            encoding
                                .get_ids()
                                .iter()
                                .copied()
                                .map(TokenId::new)
                                .collect::<Vec<_>>()
                                .into_boxed_slice()
                        })
                    })
                };
                Entry { text, tokens }
            })
            .collect();
        let retained_bytes =
            entries
                .iter()
                .try_fold(std::mem::size_of_val(entries.as_ref()), |bytes, entry| {
                    bytes.checked_add(
                        entry
                            .tokens
                            .as_ref()
                            .map_or(0, |tokens| std::mem::size_of_val(tokens.as_ref())),
                    )
                });
        debug!(
            elapsed_us = started.elapsed().as_micros() as u64,
            retained_bytes = ?retained_bytes,
            marker_count = entries.len(),
            "Compiled immutable completion marker tokens"
        );
        Self { entries }
    }
    pub(super) fn get(&self, text: &str) -> Option<&[TokenId]> {
        self.entries
            .iter()
            .find(|entry| entry.text == text)?
            .tokens
            .as_deref()
    }
}

#[cfg(test)]
mod tests;
