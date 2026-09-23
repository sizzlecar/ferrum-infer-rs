use super::*;
use crate::{
    parse_reasoning_response_for_prompt, should_defer_model_reasoning_stream_delta,
    ModelOutputProtocol,
};

fn boundaries(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
        .collect()
}
fn cases() -> Vec<String> {
    let mut cases = vec![
        "",
        "answer 🦀",
        "<think>",
        "<think>reason</think>\r\nanswer",
        "reason</think>\nanswer",
        "before<think>推理</think>\n答案",
        "hello</think>middle<think>new reasoning</think>last",
        "<think>reason</think>The literal <think> tag remains content.",
        "</think>\n{\"text\":\"<think>literal</think>\"}",
        "<thi",
        "</thi",
        "\r\n<think>x</think>\n",
        "reason\r\n",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    // All marker orderings and empty/nonempty gaps, including closer before
    // opener. These exercise actual classification transitions, not a model ID.
    for first in ["<think>", "</think>"] {
        for second in ["<think>", "</think>"] {
            for middle in ["", "中", "\r\n"] {
                cases.push(format!("prefix{first}{middle}{second}suffix"));
            }
        }
    }
    cases
}

#[test]
fn text_view_matches_existing_owned_parser_at_every_unicode_prefix() {
    for raw in cases() {
        for opened in [false, true] {
            for end in boundaries(&raw) {
                let raw = &raw[..end];
                let expected = parse_reasoning_response_for_prompt(raw, opened);
                let view = parse_text_reasoning_view(raw, opened);
                assert_eq!(
                    view.content_parts().concat(),
                    expected.content,
                    "{raw:?}, opened={opened}"
                );
                assert_eq!(
                    view.reasoning(),
                    expected.reasoning.as_deref(),
                    "{raw:?}, opened={opened}"
                );
                assert!(view.content_bytes() + view.reasoning().map_or(0, str::len) <= raw.len());
            }
        }
    }
}

fn emitted_at_prefixes(
    raw: &str,
    opened: bool,
    prefixes: impl IntoIterator<Item = usize>,
) -> usize {
    let mut sent = [0; 2];
    let mut emitted = 0;
    for (end, terminal) in prefixes
        .into_iter()
        .map(|end| (end, false))
        .chain(std::iter::once((raw.len(), true)))
    {
        let prefix = &raw[..end];
        if !terminal && should_defer_model_reasoning_stream_delta(ModelOutputProtocol::Text, prefix)
        {
            continue;
        }
        let view = parse_text_reasoning_view(prefix, opened);
        let content = view.content_parts().concat();
        for (channel, text) in [content.as_str(), view.reasoning().unwrap_or("")]
            .into_iter()
            .enumerate()
        {
            // The exact legacy stream_text_delta watermark rule, including
            // resets on shortening/non-boundary offsets after reclassification.
            if sent[channel] <= text.len() && text.is_char_boundary(sent[channel]) {
                emitted += text.len() - sent[channel];
            }
            sent[channel] = text.len();
        }
    }
    emitted
}

#[test]
fn text_view_four_epoch_emission_bound_covers_reclassification_and_terminal_flush() {
    let mut witnessed_repeated_classification = false;
    for raw in cases() {
        let offsets = boundaries(&raw);
        for opened in [false, true] {
            let mut check = |ends: Vec<usize>| {
                let emitted = emitted_at_prefixes(&raw, opened, ends);
                witnessed_repeated_classification |= emitted > raw.len();
                assert!(
                    emitted <= TEXT_REASONING_MAX_PROJECTION_EPOCHS * raw.len(),
                    "{raw:?}, opened={opened}, emitted={emitted}"
                );
            };
            check(offsets.clone());
            for (left, &first) in offsets.iter().enumerate() {
                for &second in &offsets[left..] {
                    check(vec![first, second, raw.len()]);
                }
            }
        }
    }
    assert!(
        witnessed_repeated_classification,
        "fixture must expose why a single raw-length budget is insufficient"
    );
}
