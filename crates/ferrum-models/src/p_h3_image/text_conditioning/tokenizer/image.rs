//! One original Comfy image block, followed by raw prompt text. This is
//! presentation metadata; it neither invents image features nor enables edit
//! execution without the separately compiled visual and text components.
use super::*;
use crate::{
    p_h3_image::PH3ImageModality,
    qwen3_vl_text::{Qwen3VLTextDeepStack, Qwen3VLTextVisualRun},
    qwen3_vl_vision::Qwen3VLVisionGrid,
};

mod inputs;
mod instruction;
pub const PH3_IMAGE_EDIT_PRESENTATION_PROFILE: &str =
    "p_h3_image.comfy.single_image.label_vision_raw_prompt.v1";
pub const PH3_IMAGE_EDIT_ROTARY_PROFILE: &str =
    "p_h3_image.comfy.single_image.interleaved_thw.f32_libm-0.2.15.v1";

/// One image's expanded rows. The vision markers are real text embeddings;
/// only the enclosed rows receive main and DeepStack image features. The
/// entire block, including both markers, carries the DiT Image modality tag.
pub struct PH3ImageEditPrompt {
    text: PH3ImageTextPromptBatch,
    visual: Qwen3VLTextDeepStack,
    tags: Vec<PH3ImageModality>,
    images: Vec<PromptImage>,
}
struct PromptImage {
    grid: Qwen3VLVisionGrid,
    start: usize,
    merged_height: usize,
    merged_width: usize,
}
impl PH3ImageTextTokenizer {
    pub fn prepare_single_image(
        &self,
        prompt: &str,
        grid: &Qwen3VLVisionGrid,
        merge_size: u64,
    ) -> Result<PH3ImageEditPrompt> {
        self.prepare_images(prompt, std::slice::from_ref(grid), merge_size)
    }

    /// Ordered native Picture blocks, followed by the edit instruction.
    pub fn prepare_images(
        &self,
        prompt: &str,
        grids: &[Qwen3VLVisionGrid],
        merge_size: u64,
    ) -> Result<PH3ImageEditPrompt> {
        // The pinned Comfy mRoPE position builder uses grid//2. Reject another
        // architecture's merge factor instead of silently changing that rule.
        ensure!(
            (1..=8).contains(&grids.len())
                && merge_size == 2
                && grids.iter().all(|grid| grid.temporal() == 1
                    && grid.height() > 0
                    && grid.width() > 0
                    && grid.height() % merge_size == 0
                    && grid.width() % merge_size == 0),
            "Comfy image presentation requires 1..=8 positive grids with spatial merge 2"
        );
        ensure!(
            prompt.len() <= self.limits.max_prompt_bytes && !prompt.contains("embedding:"),
            "image prompt exceeds raw-text limits or declares unsupported external embeddings"
        );
        let marker = |name| -> Result<u32> {
            let token = self
                .tokenizer
                .token_to_id(name)
                .context("missing image marker")?;
            let encoded = self
                .tokenizer
                .encode(name, false)
                .map_err(|e| anyhow::anyhow!("image marker encoding: {e}"))?;
            ensure!(
                encoded.get_ids() == [token] && u64::from(token) < self.vocabulary,
                "image marker must be one original in-vocabulary token"
            );
            Ok(token)
        };
        let begin = marker("<|vision_start|>")?;
        let end = marker("<|vision_end|>")?;
        ensure!(
            begin != end && begin != self.pad && end != self.pad,
            "image markers and placeholder PAD must be distinct"
        );
        let encode = |text: &str| -> Result<Vec<u32>> {
            let encoded = self
                .tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("raw image text encoding: {e}"))?;
            ensure!(
                !encoded
                    .get_ids()
                    .iter()
                    .any(|id| self.forbidden_vision.contains(id)),
                "raw image text cannot inject another vision block"
            );
            Ok(encoded.get_ids().to_vec())
        };
        let prompt = encode(prompt)?;
        let mut ids = Vec::new();
        let mut images = Vec::with_capacity(grids.len());
        let mut runs = Vec::with_capacity(grids.len());
        let mut tags = Vec::new();
        for (index, grid) in grids.iter().enumerate() {
            let label = encode(&format!("<Picture {}>: ", index + 1))?;
            ensure!(
                !label.is_empty(),
                "image label cannot disappear during tokenization"
            );
            let height = usize::try_from(grid.height() / merge_size)?;
            let width = usize::try_from(grid.width() / merge_size)?;
            let rows = height
                .checked_mul(width)
                .context("image feature rows overflow")?;
            let count = ids
                .len()
                .checked_add(label.len())
                .and_then(|n| n.checked_add(2))
                .and_then(|n| n.checked_add(rows))
                .and_then(|n| n.checked_add(prompt.len()))
                .context("image token count overflows")?;
            ensure!(
                count <= self.limits.max_tokens_per_prompt as usize,
                "expanded images and prompt exceed token capacity; truncation is disabled"
            );
            ids.extend(label);
            tags.resize(ids.len(), PH3ImageModality::Text);
            ids.push(begin);
            let start = ids.len();
            ids.resize(start + rows, self.pad);
            ids.push(end);
            tags.resize(ids.len(), PH3ImageModality::Image);
            runs.push(Qwen3VLTextVisualRun::new(start as u64, rows as u64)?);
            images.push(PromptImage {
                grid: grid.clone(),
                start,
                merged_height: height,
                merged_width: width,
            });
        }
        ids.extend(prompt);
        tags.resize(ids.len(), PH3ImageModality::Text);
        let text = PH3ImageTextPromptBatch::from_rows(vec![ids], self.pad, self.vocabulary)?;
        let visual = Qwen3VLTextDeepStack::new(runs)?;
        Ok(PH3ImageEditPrompt {
            text,
            visual,
            tags,
            images,
        })
    }
}
impl PH3ImageEditPrompt {
    pub fn text(&self) -> &PH3ImageTextPromptBatch {
        &self.text
    }
    pub fn visual_rows(&self) -> &Qwen3VLTextDeepStack {
        &self.visual
    }
    pub fn token_tags(&self) -> &[PH3ImageModality] {
        &self.tags
    }
    pub fn grid(&self) -> &Qwen3VLVisionGrid {
        &self.images[0].grid
    }
    pub fn grids(&self) -> impl ExactSizeIterator<Item = &Qwen3VLVisionGrid> {
        self.images.iter().map(|image| &image.grid)
    }
    /// Coordinates follow the original merged grid. The next text token
    /// resumes after the largest spatial axis, not after the flattened image.
    pub fn position_ids(&self, row: usize) -> Result<[u32; 3]> {
        ensure!(row < self.text.ids.len(), "image token row is out of range");
        let mut removed = 0;
        for image in &self.images {
            if row < image.start {
                break;
            }
            let count = image.merged_height * image.merged_width;
            if row < image.start + count {
                let local = row - image.start;
                let base = image.start - removed;
                return Ok([
                    base,
                    base + local / image.merged_width,
                    base + local % image.merged_width,
                ]
                .map(|p| p as u32));
            }
            removed += count - image.merged_height.max(image.merged_width);
        }
        Ok([(row - removed) as u32; 3])
    }
}

#[cfg(test)]
mod tests;
