//! Ordered labels and per-time-block visual slots, retaining each original
//! vision output as one source even when text timestamps separate its rows.
use super::*;

fn visual<'a>(r: &FastH3PresentationReference<'a>) -> Option<&'a Qwen3VLVisionConfig> {
    match r {
        FastH3PresentationReference::Image(v)
        | FastH3PresentationReference::Video { vision: v, .. } => Some(*v),
        FastH3PresentationReference::Audio => None,
    }
}

pub(super) fn build(
    tokenizer: &Tokenizer,
    prompt: &str,
    references: &[FastH3PresentationReference<'_>],
    limits: FastH3PresentationLimits,
) -> Result<FastH3MediaPresentation> {
    ensure!(
        limits.maximum_prompt_bytes > 0
            && limits.maximum_tokens > 0
            && limits.maximum_tokens <= 1 << 24
            && prompt.len() <= limits.maximum_prompt_bytes
            && !references.is_empty()
            && references.len() <= limits.maximum_tokens as usize,
        "presentation exceeds explicit text capacity"
    );
    ensure!(
        tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none(),
        "presentation requires tokenizer without padding or truncation"
    );
    let architecture = references
        .iter()
        .find_map(visual)
        .context("Ref2VA presentation requires a visual reference")?
        .architecture();
    for reference in references {
        if let Some(v) = visual(reference) {
            ensure!(
                v.grids().len() == 1
                    && v.architecture() == architecture
                    && architecture.deepstack_layers().len() == 3,
                "each visual reference requires one grid and the same three-DeepStack architecture"
            );
            match reference {
                FastH3PresentationReference::Image(_) => ensure!(
                    v.grids()[0].temporal() == 1,
                    "image reference requires a T1 grid"
                ),
                FastH3PresentationReference::Video {
                    block_timestamps, ..
                } => ensure!(
                    v.grids()[0].temporal() == block_timestamps.len() as u64
                        && block_timestamps.iter().all(|t| t.is_finite() && *t >= 0.)
                        && block_timestamps.windows(2).all(|t| t[0] < t[1]),
                    "video grid must match ordered finite block timestamps"
                ),
                FastH3PresentationReference::Audio => unreachable!(),
            }
        }
    }
    let marker = |name: &str| -> Result<u32> {
        let id = tokenizer
            .token_to_id(name)
            .with_context(|| format!("tokenizer lacks {name}"))?;
        ensure!(
            tokenizer.id_to_token(id).as_deref() == Some(name),
            "ambiguous visual marker"
        );
        Ok(id)
    };
    let start = marker("<|vision_start|>")?;
    let end = marker("<|vision_end|>")?;
    let image_pad = references
        .iter()
        .any(|r| matches!(r, FastH3PresentationReference::Image(_)))
        .then(|| marker("<|image_pad|>"))
        .transpose()?;
    let video_pad = references
        .iter()
        .any(|r| matches!(r, FastH3PresentationReference::Video { .. }))
        .then(|| marker("<|video_pad|>"))
        .transpose()?;
    let markers = [Some(start), Some(end), image_pad, video_pad]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    ensure!(
        markers
            .iter()
            .enumerate()
            .all(|(i, id)| !markers[..i].contains(id)),
        "visual markers must be distinct"
    );
    let raw = |text: &str| -> Result<Vec<u32>> {
        let ids = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        ensure!(
            !ids.iter().any(|id| {
                [
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                ]
                .iter()
                .any(|name| tokenizer.token_to_id(name) == Some(*id))
            }),
            "raw text introduces an unsupported visual slot"
        );
        Ok(ids)
    };
    let prompt = raw(prompt)?;
    let mut b = Builder {
        tokens: vec![],
        tags: vec![],
        spans: vec![],
        limit: limits.maximum_tokens as usize,
        reserved_prompt: prompt.len(),
        start,
        end,
    };
    let mut sources = Vec::new();
    let (mut images, mut videos, mut audios) = (0, 0, 0);
    for reference in references {
        if matches!(
            reference,
            FastH3PresentationReference::Audio
                | FastH3PresentationReference::Video {
                    has_audio: true,
                    ..
                }
        ) {
            audios += 1;
            b.text(raw(&format!("<Audio {audios}>: "))?)?;
        }
        let first_span = b.spans.len();
        match reference {
            FastH3PresentationReference::Audio => continue,
            FastH3PresentationReference::Image(v) => {
                images += 1;
                b.text(raw(&format!("<Picture {images}>: "))?)?;
                b.vision(v, image_pad.unwrap())?;
            }
            FastH3PresentationReference::Video {
                vision,
                block_timestamps,
                ..
            } => {
                videos += 1;
                b.text(raw(&format!("<Video {videos}>: "))?)?;
                for timestamp in *block_timestamps {
                    b.text(raw(&format!("<{timestamp:.1} seconds>"))?)?;
                    b.vision(vision, video_pad.unwrap())?;
                }
            }
        }
        sources.push(VisualSource {
            first_span,
            span_count: b.spans.len() - first_span,
            rows: visual(reference).unwrap().merged_rows()?,
        });
    }
    b.reserved_prompt = 0;
    if !prompt.is_empty() {
        b.text(prompt)?;
    }
    Ok(FastH3MediaPresentation {
        tokens: b.tokens,
        tags: b.tags,
        spans: b.spans,
        sources,
        visual_width: architecture.output_size(),
    })
}

struct Builder {
    tokens: Vec<u32>,
    tags: Vec<FastH3TextTagRun>,
    spans: Vec<Qwen3VLTextImageSpan>,
    limit: usize,
    reserved_prompt: usize,
    start: u32,
    end: u32,
}
impl Builder {
    fn capacity(&self, additional: usize) -> Result<()> {
        ensure!(
            self.tokens
                .len()
                .checked_add(additional)
                .and_then(|n| n.checked_add(self.reserved_prompt))
                .is_some_and(|n| n <= self.limit),
            "complete media presentation exceeds token capacity"
        );
        Ok(())
    }
    fn text(&mut self, tokens: Vec<u32>) -> Result<()> {
        ensure!(!tokens.is_empty(), "reference label must tokenize");
        self.capacity(tokens.len())?;
        self.tags.push(FastH3TextTagRun {
            rows: tokens.len() as u64,
            tag: FastH3TextTokenTag::Text,
        });
        self.tokens.extend(tokens);
        Ok(())
    }
    fn vision(&mut self, vision: &Qwen3VLVisionConfig, pad: u32) -> Result<()> {
        let grid = &vision.grids()[0];
        let span = Qwen3VLTextImageSpan::new(
            self.tokens.len() as u64 + 1,
            grid.height(),
            grid.width(),
            vision.architecture().merge_size(),
        )?;
        let count = usize::try_from(span.visual_run().length())?;
        self.capacity(count.checked_add(2).context("visual extent overflow")?)?;
        self.tags.push(FastH3TextTagRun {
            rows: count as u64 + 2,
            tag: FastH3TextTokenTag::Vision,
        });
        self.tokens.push(self.start);
        self.tokens.extend(std::iter::repeat_n(pad, count));
        self.tokens.push(self.end);
        self.spans.push(span);
        Ok(())
    }
}
