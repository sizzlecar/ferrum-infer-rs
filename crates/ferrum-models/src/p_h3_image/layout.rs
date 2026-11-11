//! P-H3 single-frame packing and rotary positions, following
//! `p_h3_image/before_denoise.py` at Diffusers
//! `c55ec27943462eeeb65c177d6ff7d8272d21d7a7`.
//!
//! Every request forms one fully bidirectional attention document. References
//! are part of that document; different requests must remain separate segments.
//! These borrowed descriptors allocate no tensor backing or runtime accounts.

use std::ops::Range;

use anyhow::{ensure, Context, Result};

use super::PH3ImageTransformerConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PH3ImageModality {
    Image = 0,
    Text = 1,
    Audio = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImageReferenceImage {
    latent_height: usize,
    latent_width: usize,
}

impl PH3ImageReferenceImage {
    pub fn new(latent_height: usize, latent_width: usize) -> Result<Self> {
        ensure!(
            latent_height > 0 && latent_width > 0,
            "reference latent dimensions must be positive"
        );
        Ok(Self {
            latent_height,
            latent_width,
        })
    }
    pub fn latent_height(self) -> usize {
        self.latent_height
    }
    pub fn latent_width(self) -> usize {
        self.latent_width
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageRowKind {
    Text { token: usize },
    ReferenceImage { image: usize, patch: usize },
    Audio { channel: usize, latent: usize },
    TargetImage { patch: usize },
}

/// `[text | reference image patches | generated audio | target image patches]`.
/// An empty reference slice denotes text-to-image. Text tags also identify the
/// vision rows embedded by Qwen3-VL; they retain their text-prefix positions.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageRequestLayout<'a> {
    text_tags: &'a [PH3ImageModality],
    references: &'a [PH3ImageReferenceImage],
    target: PH3ImageReferenceImage,
    patch_h: usize,
    patch_w: usize,
    condition_rows: usize,
    target_rows: usize,
    rows: usize,
}

impl<'a> PH3ImageRequestLayout<'a> {
    /// Latent dimensions, after VAE compression and canvas rounding, are
    /// explicit. This constructor never silently resizes a request.
    pub fn new(
        config: &PH3ImageTransformerConfig,
        text_tags: &'a [PH3ImageModality],
        target: PH3ImageReferenceImage,
        references: &'a [PH3ImageReferenceImage],
    ) -> Result<Self> {
        ensure!(
            !text_tags.is_empty(),
            "P-H3 text conditioning must contain a sink token"
        );
        ensure!(
            text_tags.iter().all(|tag| *tag != PH3ImageModality::Audio),
            "P-H3 text prefix supports text and vision tags only"
        );
        ensure!(
            references.len() <= 8,
            "P-H3-Image supports at most eight reference images"
        );
        let [patch_t, patch_h, patch_w] = config.patch_size();
        ensure!(
            patch_t == 1,
            "P-H3-Image layout requires single-frame patches"
        );
        let patch_h = usize::try_from(patch_h)?;
        let patch_w = usize::try_from(patch_w)?;
        let target_rows = patch_rows(target, patch_h, patch_w)?;
        let condition_rows = references.iter().try_fold(0_usize, |sum, reference| {
            sum.checked_add(patch_rows(*reference, patch_h, patch_w)?)
                .context("reference patch count overflows")
        })?;
        let rows = text_tags
            .len()
            .checked_add(condition_rows)
            .and_then(|rows| rows.checked_add(Self::AUDIO_ROWS))
            .and_then(|rows| rows.checked_add(target_rows))
            .context("P-H3 packed sequence overflows")?;
        checked_elements::<f64>(&[rows, 3])?;
        Ok(Self {
            text_tags,
            references,
            target,
            patch_h,
            patch_w,
            condition_rows,
            target_rows,
            rows,
        })
    }

    // These are model semantics, not capacity limits: P-H3 keeps two audio
    // latents per stereo channel and discards them after image denoising.
    pub const AUDIO_CHANNELS: usize = 2;
    pub const AUDIO_LATENTS_PER_CHANNEL: usize = 2;
    pub const AUDIO_ROWS: usize = Self::AUDIO_CHANNELS * Self::AUDIO_LATENTS_PER_CHANNEL;

    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn text_rows(self) -> Range<usize> {
        0..self.text_tags.len()
    }
    pub fn reference_rows(self) -> Range<usize> {
        self.text_tags.len()..self.audio_rows().start
    }
    pub fn audio_rows(self) -> Range<usize> {
        let start = self.text_tags.len() + self.condition_rows;
        start..start + Self::AUDIO_ROWS
    }
    pub fn target_rows(self) -> Range<usize> {
        self.audio_rows().end..self.rows
    }
    pub fn image_row_count(self) -> usize {
        self.condition_rows + self.target_rows
    }
    pub fn image_row(self, ordinal: usize) -> Option<usize> {
        if ordinal < self.condition_rows {
            Some(self.text_tags.len() + ordinal)
        } else if ordinal < self.image_row_count() {
            Some(self.target_rows().start + ordinal - self.condition_rows)
        } else {
            None
        }
    }

    pub fn row_kind(self, row: usize) -> Option<PH3ImageRowKind> {
        if row >= self.rows {
            return None;
        }
        if row < self.text_tags.len() {
            return Some(PH3ImageRowKind::Text { token: row });
        }
        let mut cursor = self.text_tags.len();
        for (image, reference) in self.references.iter().enumerate() {
            let count =
                (reference.latent_height / self.patch_h) * (reference.latent_width / self.patch_w);
            if row < cursor + count {
                return Some(PH3ImageRowKind::ReferenceImage {
                    image,
                    patch: row - cursor,
                });
            }
            cursor += count;
        }
        if self.audio_rows().contains(&row) {
            let index = row - self.audio_rows().start;
            Some(PH3ImageRowKind::Audio {
                channel: index / Self::AUDIO_LATENTS_PER_CHANNEL,
                latent: index % Self::AUDIO_LATENTS_PER_CHANNEL,
            })
        } else {
            Some(PH3ImageRowKind::TargetImage {
                patch: row - self.target_rows().start,
            })
        }
    }

    pub fn modality(self, row: usize) -> Option<PH3ImageModality> {
        self.row_kind(row).map(|kind| match kind {
            PH3ImageRowKind::Text { token } => self.text_tags[token],
            PH3ImageRowKind::Audio { .. } => PH3ImageModality::Audio,
            _ => PH3ImageModality::Image,
        })
    }

    /// Original double-precision layout. The rotary adapter later casts each
    /// coordinate to FP32 before multiplying by its inverse frequencies.
    pub fn position(self, row: usize) -> Option<[f64; 3]> {
        let origin = self.text_tags.len() as f64;
        self.row_kind(row).map(|kind| match kind {
            PH3ImageRowKind::Text { token } => [token as f64, 0.0, 0.0],
            PH3ImageRowKind::ReferenceImage { image, patch } => {
                let spatial =
                    frame_position(self.references[image], self.patch_h, self.patch_w, patch);
                [origin + image as f64, spatial[0], spatial[1]]
            }
            PH3ImageRowKind::Audio { channel, latent } => {
                let width_count = self.target.latent_width / self.patch_w;
                let width_index = if channel == 0 { 0 } else { width_count - 1 };
                let width = spatial_coordinate(
                    self.target.latent_width,
                    self.patch_w,
                    area_sqrt(self.target),
                    width_index,
                );
                [
                    origin + self.references.len() as f64 + latent as f64,
                    0.0,
                    width,
                ]
            }
            PH3ImageRowKind::TargetImage { patch } => {
                let spatial = frame_position(self.target, self.patch_h, self.patch_w, patch);
                [
                    origin + self.references.len() as f64,
                    spatial[0],
                    spatial[1],
                ]
            }
        })
    }

    /// Preserve the packed sequence's per-row time semantics. Prefix text
    /// inherits image time, including vision-tagged prefix rows. Reference
    /// image rows use max(image_time, conditioning_noise_aug).
    pub fn write_row_timesteps(
        self,
        image_time: f32,
        audio_time: f32,
        conditioning_noise_aug: f32,
        output: &mut [f32],
    ) -> Result<()> {
        ensure!(
            output.len() == self.rows,
            "P-H3 row timestep destination length differs"
        );
        ensure!(
            [image_time, audio_time, conditioning_noise_aug]
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "P-H3 row timesteps must be finite and within [0,1]"
        );
        for (row, value) in output.iter_mut().enumerate() {
            *value = match self.row_kind(row).unwrap() {
                PH3ImageRowKind::ReferenceImage { .. } => image_time.max(conditioning_noise_aug),
                PH3ImageRowKind::Audio { .. } => audio_time,
                _ => image_time,
            };
        }
        Ok(())
    }

    pub fn write_positions(self, output: &mut [[f64; 3]]) -> Result<()> {
        ensure!(
            output.len() == self.rows,
            "P-H3 position destination length differs"
        );
        for (row, position) in output.iter_mut().enumerate() {
            *position = self.position(row).unwrap();
        }
        Ok(())
    }
}

/// Packed requests with disjoint full-attention segments, ready to describe a
/// segmented attention operation. This shape proof alone does not execute or
/// authorize attention, nor share prefix KV across H3 denoising iterations.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImagePackedBatchLayout<'a> {
    requests: &'a [PH3ImageRequestLayout<'a>],
    rows: usize,
}

impl<'a> PH3ImagePackedBatchLayout<'a> {
    pub fn new(requests: &'a [PH3ImageRequestLayout<'a>]) -> Result<Self> {
        ensure!(
            !requests.is_empty(),
            "P-H3 packed batch must contain a request"
        );
        let rows = requests.iter().try_fold(0_usize, |sum, request| {
            sum.checked_add(request.rows)
                .context("P-H3 packed batch row count overflows")
        })?;
        checked_elements::<f64>(&[rows, 3])?;
        Ok(Self { requests, rows })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn request_count(self) -> usize {
        self.requests.len()
    }
    pub fn request_range(self, request: usize) -> Option<Range<usize>> {
        let layout = self.requests.get(request)?;
        let start = self.requests[..request]
            .iter()
            .map(|layout| layout.rows)
            .sum::<usize>();
        Some(start..start + layout.rows)
    }
    pub fn request_at_row(self, row: usize) -> Option<(usize, usize)> {
        if row >= self.rows {
            return None;
        }
        let mut start = 0;
        for (request, layout) in self.requests.iter().enumerate() {
            if row < start + layout.rows {
                return Some((request, row - start));
            }
            start += layout.rows;
        }
        None
    }
    pub fn allows_attention(self, query: usize, key: usize) -> bool {
        match (self.request_at_row(query), self.request_at_row(key)) {
            (Some((request, _)), Some((other, _))) => request == other,
            _ => false,
        }
    }
    pub fn write_segment_lengths(self, output: &mut [u64]) -> Result<()> {
        ensure!(
            output.len() == self.requests.len(),
            "P-H3 segment length destination differs"
        );
        // usize fits u64 on supported Rust targets. No partial write can fail.
        for (value, request) in output.iter_mut().zip(self.requests) {
            *value = request.rows as u64;
        }
        Ok(())
    }
    pub fn write_positions(self, output: &mut [[f64; 3]]) -> Result<()> {
        ensure!(
            output.len() == self.rows,
            "P-H3 packed positions destination differs"
        );
        let mut remaining = output;
        for request in self.requests {
            let (current, rest) = remaining.split_at_mut(request.rows);
            request.write_positions(current)?;
            remaining = rest;
        }
        Ok(())
    }
}

fn patch_rows(image: PH3ImageReferenceImage, patch_h: usize, patch_w: usize) -> Result<usize> {
    ensure!(
        image.latent_height % patch_h == 0 && image.latent_width % patch_w == 0,
        "P-H3 latent dimensions must divide into complete image patches"
    );
    image
        .latent_height
        .checked_mul(image.latent_width)
        .context("P-H3 latent area overflows")?;
    (image.latent_height / patch_h)
        .checked_mul(image.latent_width / patch_w)
        .context("P-H3 patch grid overflows")
}

fn area_sqrt(image: PH3ImageReferenceImage) -> f64 {
    libm::sqrt((image.latent_height * image.latent_width) as f64)
}

fn spatial_coordinate(dim: usize, patch: usize, sqrt_area: f64, index: usize) -> f64 {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0_f64 - ratio) / 2.0_f64;
    let right = left + ratio;
    let step = (right - left) / (dim / patch) as f64;
    (left + index as f64 * step) * 32.0_f64
}

fn frame_position(
    image: PH3ImageReferenceImage,
    patch_h: usize,
    patch_w: usize,
    patch: usize,
) -> [f64; 2] {
    let width = image.latent_width / patch_w;
    let sqrt_area = area_sqrt(image);
    [
        spatial_coordinate(image.latent_height, patch_h, sqrt_area, patch / width),
        spatial_coordinate(image.latent_width, patch_w, sqrt_area, patch % width),
    ]
}

pub(super) fn checked_elements<T>(dimensions: &[usize]) -> Result<usize> {
    let elements = dimensions.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .context("P-H3 tensor element count overflows")
    })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .context("P-H3 tensor byte count overflows")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "P-H3 tensor storage exceeds address space"
    );
    Ok(elements)
}

#[cfg(test)]
mod tests;
