//! Borrowed view of the existing Text reasoning parser. No allocation or
//! protocol inference is performed here; the rendered prompt supplies state.
use super::{THINK_END_TAG, THINK_START_TAG};

/// Text classification can enter, in order: no marker, closer without opener,
/// opener without its closer, and complete opener/closer. Within each epoch
/// channel lengths only grow and their disjoint slices total at most raw.len().
/// The legacy streaming length watermarks can therefore emit at most four
/// times the final raw length, including reclassification across epochs.
/// Prompt-opened parsing visits no more epochs than this general case.
pub const TEXT_REASONING_MAX_PROJECTION_EPOCHS: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct TextReasoningView<'a> {
    content: [&'a str; 2],
    reasoning: Option<&'a str>,
}

impl<'a> TextReasoningView<'a> {
    pub fn content_parts(self) -> [&'a str; 2] {
        self.content
    }
    pub fn reasoning(self) -> Option<&'a str> {
        self.reasoning
    }
    pub fn content_bytes(self) -> usize {
        self.content[0].len() + self.content[1].len()
    }
}

/// Equivalent to parse_reasoning_response_for_prompt, including Some("") for
/// an explicitly opened, unfinished empty reasoning block. Keeping views as
/// slices preserves Unicode boundaries and avoids concatenation scratch here.
pub fn parse_text_reasoning_view(text: &str, prompt_opened: bool) -> TextReasoningView<'_> {
    if prompt_opened {
        let end = text.find(THINK_END_TAG);
        let repeated_open = text
            .find(THINK_START_TAG)
            .is_some_and(|start| end.is_none_or(|end| start < end));
        if !repeated_open {
            return match end {
                None => TextReasoningView {
                    content: ["", ""],
                    reasoning: (!text.is_empty()).then_some(text),
                },
                Some(end) => TextReasoningView {
                    content: [
                        text[end + THINK_END_TAG.len()..].trim_start_matches(['\r', '\n']),
                        "",
                    ],
                    reasoning: (end != 0).then_some(&text[..end]),
                },
            };
        }
    }
    let Some(start) = text.find(THINK_START_TAG) else {
        return match text.find(THINK_END_TAG) {
            Some(end) => TextReasoningView {
                content: [
                    text[end + THINK_END_TAG.len()..].trim_start_matches(['\r', '\n']),
                    "",
                ],
                reasoning: (end != 0).then_some(&text[..end]),
            },
            None => TextReasoningView {
                content: [text, ""],
                reasoning: None,
            },
        };
    };
    let before = &text[..start];
    let after_start = &text[start + THINK_START_TAG.len()..];
    match after_start.find(THINK_END_TAG) {
        None => TextReasoningView {
            content: [before, ""],
            reasoning: Some(after_start),
        },
        Some(end) => TextReasoningView {
            content: [
                before,
                after_start[end + THINK_END_TAG.len()..].trim_start_matches(['\r', '\n']),
            ],
            reasoning: (end != 0).then_some(&after_start[..end]),
        },
    }
}

#[cfg(test)]
mod tests;
