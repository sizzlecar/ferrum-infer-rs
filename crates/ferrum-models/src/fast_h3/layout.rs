use std::ops::Range;

use anyhow::{ensure, Context, Result};
use ferrum_diffusion::SpatiotemporalLatentShape;
use serde::{Deserialize, Serialize};

/// H3's causal VAE bucket geometry. No product duration or memory default is
/// chosen here. Requested and aligned frames remain separately observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3VideoGeometry {
    requested_frames: usize,
    aligned_frames: usize,
    latent_shape: SpatiotemporalLatentShape,
    audio_latents_per_channel: usize,
}

impl FastH3VideoGeometry {
    pub fn new(width: usize, height: usize, requested_frames: usize) -> Result<Self> {
        ensure!(
            requested_frames > 0 && width > 0 && height > 0,
            "video geometry must be positive"
        );
        ensure!(
            width % 32 == 0 && height % 32 == 0,
            "H3 canvas dimensions must be multiples of 32"
        );
        let padding = (5 + 17 - requested_frames % 17) % 17;
        let aligned_frames = requested_frames
            .checked_add(padding)
            .context("aligned frame count overflow")?;
        let latent_frames = ((aligned_frames - 5) / 17)
            .checked_mul(5)
            .and_then(|value| value.checked_add(2))
            .context("latent frame count overflow")?;
        // round(frames / 24 * 40) = round(5*frames/3); this rational has
        // no half-integer ties, so integer arithmetic also avoids float drift.
        let audio_latents_per_channel = aligned_frames
            .checked_mul(5)
            .and_then(|value| value.checked_add(1))
            .context("audio frame count overflow")?
            / 3;
        let latent_height = height / 16;
        let latent_width = width / 16;
        let latent_shape = visual_shape(latent_frames, latent_height, latent_width)?;
        Ok(Self {
            requested_frames,
            aligned_frames,
            latent_shape,
            audio_latents_per_channel,
        })
    }
    pub fn requested_frames(&self) -> usize {
        self.requested_frames
    }
    pub fn aligned_frames(&self) -> usize {
        self.aligned_frames
    }
    pub fn latent_frames(&self) -> usize {
        self.latent_shape.ncthw_dims()[2]
    }
    pub fn latent_height(&self) -> usize {
        self.latent_shape.ncthw_dims()[3]
    }
    pub fn latent_width(&self) -> usize {
        self.latent_shape.ncthw_dims()[4]
    }
    pub fn latent_shape(&self) -> SpatiotemporalLatentShape {
        self.latent_shape
    }
    pub fn audio_latents_per_channel(&self) -> usize {
        self.audio_latents_per_channel
    }
    pub fn video_rows(&self) -> usize {
        self.latent_shape.tokens_per_batch()
    }
}

/// Prepared reference geometry, after its own encoder has produced latents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FastH3Reference {
    Image {
        latent_height: usize,
        latent_width: usize,
    },
    Audio {
        latents_per_channel: usize,
    },
    Video {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_latents_per_channel: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3SegmentKind {
    Text,
    ReferenceImage,
    ReferenceVideo,
    ReferenceAudio,
    TargetAudio,
    TargetVideo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3Segment {
    pub kind: FastH3SegmentKind,
    pub packed_rows: Range<usize>,
}

/// Ordered document spans. These are not an attention mask: the family must
/// additionally materialize positions/tags and trained VSA regions. Independent
/// documents in a batch require explicit attention isolation.
#[derive(Clone, Debug)]
pub struct FastH3PackedLayout {
    segments: Vec<FastH3Segment>,
    sequence_rows: usize,
    condition_video_rows: usize,
    condition_audio_rows: usize,
    target_video_rows: Range<usize>,
    target_audio_rows: Range<usize>,
}

impl FastH3PackedLayout {
    pub fn ref2va(
        text_rows: usize,
        references: &[FastH3Reference],
        target: &FastH3VideoGeometry,
        max_sequence_rows: usize,
    ) -> Result<Self> {
        ensure!(
            text_rows > 0 && !references.is_empty(),
            "FastH3 Ref2VA needs text rows and at least one reference"
        );
        let mut layout = Self {
            segments: Vec::new(),
            sequence_rows: 0,
            condition_video_rows: 0,
            condition_audio_rows: 0,
            target_video_rows: 0..0,
            target_audio_rows: 0..0,
        };
        layout.push(FastH3SegmentKind::Text, text_rows, max_sequence_rows)?;
        for reference in references {
            match *reference {
                FastH3Reference::Image {
                    latent_height,
                    latent_width,
                } => {
                    let rows = visual_rows(1, latent_height, latent_width)?;
                    layout.push(FastH3SegmentKind::ReferenceImage, rows, max_sequence_rows)?;
                    layout.condition_video_rows += rows;
                }
                FastH3Reference::Audio {
                    latents_per_channel,
                } => layout.push_reference_audio(latents_per_channel, max_sequence_rows)?,
                FastH3Reference::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    audio_latents_per_channel,
                } => {
                    if let Some(count) = audio_latents_per_channel {
                        layout.push_reference_audio(count, max_sequence_rows)?;
                    }
                    let rows = visual_rows(latent_frames, latent_height, latent_width)?;
                    layout.push(FastH3SegmentKind::ReferenceVideo, rows, max_sequence_rows)?;
                    layout.condition_video_rows += rows;
                }
            }
        }
        let audio_rows = stereo_rows(target.audio_latents_per_channel)?;
        layout.push(
            FastH3SegmentKind::TargetAudio,
            audio_rows,
            max_sequence_rows,
        )?;
        layout.push(
            FastH3SegmentKind::TargetVideo,
            target.video_rows(),
            max_sequence_rows,
        )?;
        // Ranges in the respective modality buffers, not the packed sequence.
        layout.target_audio_rows =
            layout.condition_audio_rows..layout.condition_audio_rows + audio_rows;
        layout.target_video_rows =
            layout.condition_video_rows..layout.condition_video_rows + target.video_rows();
        Ok(layout)
    }
    fn push(&mut self, kind: FastH3SegmentKind, rows: usize, capacity: usize) -> Result<()> {
        let end = self
            .sequence_rows
            .checked_add(rows)
            .context("H3 packed row count overflow")?;
        ensure!(
            rows > 0 && end <= capacity,
            "H3 packed rows exceed capacity or contain an empty segment"
        );
        self.segments.push(FastH3Segment {
            kind,
            packed_rows: self.sequence_rows..end,
        });
        self.sequence_rows = end;
        Ok(())
    }
    fn push_reference_audio(&mut self, frames: usize, capacity: usize) -> Result<()> {
        let rows = stereo_rows(frames)?;
        self.push(FastH3SegmentKind::ReferenceAudio, rows, capacity)?;
        self.condition_audio_rows += rows;
        Ok(())
    }
    pub fn segments(&self) -> &[FastH3Segment] {
        &self.segments
    }
    pub fn sequence_rows(&self) -> usize {
        self.sequence_rows
    }
    pub fn target_video_rows(&self) -> Range<usize> {
        self.target_video_rows.clone()
    }
    pub fn target_audio_rows(&self) -> Range<usize> {
        self.target_audio_rows.clone()
    }
}

fn stereo_rows(frames: usize) -> Result<usize> {
    ensure!(frames > 0, "audio reference needs nonempty latents");
    frames
        .checked_mul(2)
        .context("stereo audio row count overflow")
}

fn visual_rows(frames: usize, height: usize, width: usize) -> Result<usize> {
    Ok(visual_shape(frames, height, width)?.tokens_per_batch())
}

fn visual_shape(frames: usize, height: usize, width: usize) -> Result<SpatiotemporalLatentShape> {
    // This model fixes 24 latent channels and [1,2,2] patches. The shared
    // contract keeps time distinct and checks the complete storage extent.
    Ok(SpatiotemporalLatentShape::new(
        [1, 24, frames, height, width],
        [1, 2, 2],
    )?)
}
