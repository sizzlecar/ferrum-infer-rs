//! Original processor placeholder expansion and request-local THW metadata.
//! No visual tensor or generation result is invented by tokenization. Actual
//! image features and their managed owners must be bound by the caller.
use super::*;
use crate::qwen3_vl_vision::Qwen3VLVisionGrid;

const IMAGE: &str = "<|image_pad|>";
const START: &str = "<|vision_start|>";
const END: &str = "<|vision_end|>";

#[derive(Clone, Debug)]
pub struct PH3ImagePromptExpansionImageRun {
    token_start: usize,
    rows: usize,
    height: usize,
    width: usize,
    position_start: usize,
}
impl PH3ImagePromptExpansionImageRun {
    pub fn token_start(&self) -> usize {
        self.token_start
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn merged_height(&self) -> usize {
        self.height
    }
    pub fn merged_width(&self) -> usize {
        self.width
    }
}

/// Bounded tokenizer/processor metadata only. This identity is deliberately
/// insufficient for prefix reuse: an actual image/feature owner is also needed.
#[derive(Clone, Debug)]
pub struct PH3ImagePromptExpansionImageTokens {
    tokens: Vec<u32>,
    runs: Vec<PH3ImagePromptExpansionImageRun>,
    maximum_tokens: usize,
    rope_delta: i64,
    source_sha256: [String; 3],
}
impl PH3ImagePromptExpansionImageTokens {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn image_runs(&self) -> &[PH3ImagePromptExpansionImageRun] {
        &self.runs
    }
    pub fn maximum_tokens(&self) -> usize {
        self.maximum_tokens
    }
    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }
    pub fn source_sha256(&self) -> &[String; 3] {
        &self.source_sha256
    }

    /// Actual image feature row, in the caller's ordered source images. Start
    /// and end markers retain ordinary token embeddings, as do decode tokens.
    pub fn image_row(&self, physical_token: usize) -> Option<(usize, usize)> {
        self.runs.iter().enumerate().find_map(|(image, run)| {
            physical_token
                .checked_sub(run.token_start)
                .filter(|&row| row < run.rows)
                .map(|row| (image, row))
        })
    }

    /// Position coordinates do not replace the physical cache/token frontier.
    /// Implements the fixed HF Qwen3.5 image grouping rule (T=1), including the
    /// signed continuation delta. No shared mutable model-wide delta exists.
    pub fn position(&self, physical_token: usize) -> Result<[u64; 3]> {
        if physical_token >= self.maximum_tokens {
            return Err(invalid("position exceeds the admitted generation extent"));
        }
        if physical_token >= self.tokens.len() {
            let position = i64::try_from(physical_token)
                .ok()
                .and_then(|p| p.checked_add(self.rope_delta))
                .and_then(|p| u64::try_from(p).ok())
                .ok_or_else(|| invalid("decode position overflows its signed delta"))?;
            return Ok([position; 3]);
        }
        let mut physical_end = 0;
        let mut position_end = 0;
        for run in &self.runs {
            if physical_token < run.token_start {
                return Ok([(position_end + (physical_token - physical_end)) as u64; 3]);
            }
            if physical_token < run.token_start + run.rows {
                let local = physical_token - run.token_start;
                return Ok([
                    run.position_start as u64,
                    (run.position_start + local / run.width) as u64,
                    (run.position_start + local % run.width) as u64,
                ]);
            }
            physical_end = run.token_start + run.rows;
            position_end = run.position_start + run.height.max(run.width);
        }
        Ok([(position_end + (physical_token - physical_end)) as u64; 3])
    }
}

impl PH3ImagePromptExpansionProtocol {
    pub fn encode_image_input(
        &self,
        input: &PH3ImagePromptExpansionInput,
        grids: &[Qwen3VLVisionGrid],
        spatial_merge_size: u64,
    ) -> Result<PH3ImagePromptExpansionImageTokens> {
        if input.source_sha256 != self.source_sha256
            || input.images == 0
            || usize::from(input.images) != grids.len()
            || spatial_merge_size == 0
        {
            return Err(invalid(
                "image input needs its original processor and ordered grids",
            ));
        }
        let mut dimensions = Vec::with_capacity(grids.len());
        let mut expanded_bytes = input.text.len();
        for grid in grids {
            if grid.temporal() != 1
                || grid.height() % spatial_merge_size != 0
                || grid.width() % spatial_merge_size != 0
            {
                return Err(invalid(
                    "static image grids require T=1 and complete spatial merge cells",
                ));
            }
            let height = usize::try_from(grid.height() / spatial_merge_size)
                .map_err(|_| invalid("image grid height exceeds address space"))?;
            let width = usize::try_from(grid.width() / spatial_merge_size)
                .map_err(|_| invalid("image grid width exceeds address space"))?;
            let rows = height
                .checked_mul(width)
                .filter(|&n| n > 0 && n <= self.context_tokens as usize)
                .ok_or_else(|| invalid("image tokens exceed source context"))?;
            expanded_bytes = (rows - 1)
                .checked_mul(IMAGE.len())
                .and_then(|n| expanded_bytes.checked_add(n))
                .filter(|&n| n <= self.limits.max_rendered_bytes)
                .ok_or_else(|| {
                    invalid("expanded image placeholders exceed the rendered metadata limit")
                })?;
            dimensions.push((height, width, rows));
        }
        // Extra user-supplied image markers cannot silently acquire a feature
        // row belonging to another source image or become token-only content.
        if [IMAGE, START, END]
            .iter()
            .any(|marker| input.text.matches(*marker).count() != grids.len())
        {
            return Err(invalid(
                "rendered image markers do not match the supplied images",
            ));
        }
        let special = |marker: &str| -> Result<u32> {
            let id = self
                .tokenizer
                .token_to_id(marker)
                .ok_or_else(|| invalid("missing original image special token"))?;
            let encoded = self
                .tokenizer
                .encode(marker, false)
                .map_err(|e| invalid(&e.to_string()))?;
            if encoded.get_ids() != [id] {
                return Err(invalid("image marker is not one actual tokenizer token"));
            }
            Ok(id)
        };
        let [image, start, end] = [special(IMAGE)?, special(START)?, special(END)?];
        if image == start || image == end || start == end {
            return Err(invalid("image token identities must be distinct"));
        }
        let mut expanded = String::new();
        expanded
            .try_reserve_exact(expanded_bytes)
            .map_err(|e| invalid(&e.to_string()))?;
        let mut parts = input.text.split(IMAGE);
        expanded.push_str(parts.next().expect("split has initial text"));
        for (_, _, rows) in &dimensions {
            for _ in 0..*rows {
                expanded.push_str(IMAGE);
            }
            expanded.push_str(parts.next().expect("checked placeholder count"));
        }
        let encoded = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| invalid(&e.to_string()))?;
        let tokens = encoded.get_ids();
        let maximum_tokens = tokens
            .len()
            .checked_add(input.max_new_tokens as usize)
            .filter(|&n| n <= self.context_tokens as usize)
            .ok_or_else(|| {
                invalid("expanded prompt plus complete generation exceeds the source context")
            })?;
        let starts = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == start).then_some(i))
            .collect::<Vec<_>>();
        if starts.len() != dimensions.len()
            || tokens.iter().filter(|&&id| id == end).count() != dimensions.len()
        {
            return Err(invalid(
                "encoded image boundary identities differ from original framing",
            ));
        }
        let mut runs = Vec::with_capacity(dimensions.len());
        let mut physical_end = 0usize;
        let mut position_end = 0usize;
        let mut image_rows = 0usize;
        for (marker, (height, width, rows)) in starts.into_iter().zip(dimensions) {
            let token_start = marker + 1;
            let token_end = token_start
                .checked_add(rows)
                .filter(|&e| e < tokens.len())
                .ok_or_else(|| invalid("encoded image extent exceeds prompt"))?;
            if token_start < physical_end
                || tokens[token_start..token_end].iter().any(|&id| id != image)
                || tokens[token_end] != end
            {
                return Err(invalid(
                    "encoded image feature run is not exactly bracketed by its markers",
                ));
            }
            let position_start = position_end + (token_start - physical_end);
            physical_end = token_end;
            position_end = position_start + height.max(width);
            image_rows += rows;
            runs.push(PH3ImagePromptExpansionImageRun {
                token_start,
                rows,
                height,
                width,
                position_start,
            });
        }
        if tokens.iter().filter(|&&id| id == image).count() != image_rows {
            return Err(invalid(
                "image-pad tokens exist outside the original image runs",
            ));
        }
        let position_count = position_end + (tokens.len() - physical_end);
        let rope_delta = i64::try_from(position_count)
            .ok()
            .zip(i64::try_from(tokens.len()).ok())
            .map(|(p, t)| p - t)
            .ok_or_else(|| invalid("image continuation delta exceeds signed address space"))?;
        Ok(PH3ImagePromptExpansionImageTokens {
            tokens: tokens.to_vec(),
            runs,
            maximum_tokens,
            rope_delta,
            source_sha256: self.source_sha256.clone(),
        })
    }
}

fn invalid(message: &str) -> PH3ImagePromptExpansionError {
    PH3ImagePromptExpansionError::Request(message.to_owned())
}

#[cfg(test)]
mod tests;
