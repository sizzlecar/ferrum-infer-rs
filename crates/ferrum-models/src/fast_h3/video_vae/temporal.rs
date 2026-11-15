use super::FastH3VideoVaeConfig;
use anyhow::{ensure, Context, Result};

/// Checked scalar plan for `_temporal_decode_plan` / `_decode_chunks` in the
/// pinned original VAE. No frame-sized Vec, execution state or allocator lives
/// here. The caller must retain each unfinalized overlap in its managed owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3VideoDecodePlan {
    latent_frames: usize,
    repeat_last_latents: usize,
    chunks: usize,
    frames: usize,
    chunk_latents: usize,
    overlap_latents: usize,
    ratio: usize,
    pre_padding: usize,
    overlap_frames: usize,
    main_frames: usize,
}

/// The raw decoder sees `latent_count` time steps, with out-of-range source
/// indices explicitly mapped to the final latent. Main and overlap ranges
/// index the raw `[B,3,T,H,W]` clip, not previously rounded pixel bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3VideoDecodeChunk {
    pub latent_start: usize,
    pub latent_count: usize,
    pub raw_frames: usize,
    pub main_start: usize,
    pub publish_frames: usize,
    pub blend_prefix_frames: usize,
    pub next_overlap_start: usize,
    pub next_overlap_frames: usize,
    pub output_start: usize,
}

impl FastH3VideoDecodePlan {
    pub fn new(
        config: &FastH3VideoVaeConfig,
        latent_frames: usize,
        maximum_output_frames: usize,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            latent_frames > 0,
            "video decoding requires positive latent frames"
        );
        let ratio = config.temporal_compression_ratio()?;
        let pre_padding = (ratio - config.clip_length() % ratio) % ratio;
        let chunk_latents = config
            .clip_length()
            .checked_add(ratio - 1)
            .context("clip token count overflow")?
            / ratio;
        let overlap_latents = (chunk_latents - config.token_drop() % chunk_latents) % chunk_latents;
        let overlap_frames = overlap_latents
            .checked_mul(ratio)
            .context("temporal overlap overflow")?
            .saturating_sub(pre_padding);
        let num_tokens = latent_frames
            .checked_add(config.token_drop())
            .context("latent count overflow")?;
        let mut pad = (chunk_latents - num_tokens % chunk_latents) % chunk_latents;
        let mut chunks = num_tokens
            .checked_add(pad)
            .context("padded latent count overflow")?
            / chunk_latents;
        chunks = chunks
            .checked_sub(usize::from(config.token_drop() > 0))
            .context("invalid temporal chunk count")?;
        if chunks == 0 {
            pad = pad
                .checked_add(chunk_latents)
                .context("latent padding overflow")?;
            chunks = 1;
        }
        let main_frames = chunk_latents
            .checked_mul(ratio)
            .and_then(|n| n.checked_sub(pre_padding))
            .context("main frame count overflow")?;
        let mut frames = chunks
            .checked_mul(main_frames)
            .and_then(|n| {
                n.checked_add(if config.token_drop() > 0 {
                    overlap_frames
                } else {
                    0
                })
            })
            .context("decoded frame count overflow")?;
        // Count multiples of chunk_latents in [latent_frames, latent_frames+pad).
        // This equals upstream's sum without iterating possibly huge padding.
        let end = latent_frames
            .checked_add(pad)
            .context("padded latent count overflow")?;
        let multiples = if pad == 0 {
            0
        } else {
            (end - 1) / chunk_latents - (latent_frames - 1) / chunk_latents
        };
        let intra_tail = config.clip_length() % ratio;
        let saving = if intra_tail == 0 {
            0
        } else {
            multiples
                .checked_mul(ratio - intra_tail)
                .context("tail count overflow")?
        };
        let removed = pad
            .checked_mul(ratio)
            .and_then(|n| n.checked_sub(saving))
            .context("removed frame count overflow")?;
        frames = frames
            .checked_sub(removed)
            .context("inconsistent temporal trim")?;
        ensure!(
            frames > 0 && frames <= maximum_output_frames,
            "decoded frame count exceeds output capacity or is empty"
        );
        // All descriptor arithmetic, including reads of replicated padding,
        // is checked now rather than deferred until a native submission.
        let read = chunk_latents
            .checked_add(overlap_latents)
            .context("clip read overflow")?;
        let last_end = (chunks - 1)
            .checked_mul(chunk_latents)
            .and_then(|n| n.checked_add(read))
            .context("last clip range overflow")?;
        ensure!(
            last_end <= end,
            "temporal plan reads beyond repeated padding"
        );
        read.checked_mul(ratio)
            .context("raw clip frame count overflow")?;
        Ok(Self {
            latent_frames,
            repeat_last_latents: pad,
            chunks,
            frames,
            chunk_latents,
            overlap_latents,
            ratio,
            pre_padding,
            overlap_frames,
            main_frames,
        })
    }
    pub fn latent_frames(&self) -> usize {
        self.latent_frames
    }
    pub fn repeat_last_latents(&self) -> usize {
        self.repeat_last_latents
    }
    pub fn chunk_count(&self) -> usize {
        self.chunks
    }
    pub fn output_frames(&self) -> usize {
        self.frames
    }
    pub fn overlap_frames(&self) -> usize {
        self.overlap_frames
    }
    pub fn final_overlap_frames(&self) -> usize {
        self.frames.saturating_sub(self.chunks * self.main_frames)
    }
    pub fn source_latent(&self, padded_index: usize) -> Result<usize> {
        ensure!(
            padded_index < self.latent_frames + self.repeat_last_latents,
            "latent index exceeds declared repeated padding"
        );
        Ok(padded_index.min(self.latent_frames - 1))
    }
    pub fn chunk(&self, index: usize) -> Result<FastH3VideoDecodeChunk> {
        ensure!(index < self.chunks, "temporal chunk index out of range");
        let output_start = (index * self.main_frames).min(self.frames);
        Ok(FastH3VideoDecodeChunk {
            latent_start: index * self.chunk_latents,
            latent_count: self.chunk_latents + self.overlap_latents,
            raw_frames: (self.chunk_latents + self.overlap_latents) * self.ratio,
            main_start: self.pre_padding,
            publish_frames: self.main_frames.min(self.frames - output_start),
            blend_prefix_frames: if index == 0 { 0 } else { self.overlap_frames },
            next_overlap_start: (self.chunk_latents * self.ratio + self.pre_padding)
                .min((self.chunk_latents + self.overlap_latents) * self.ratio),
            next_overlap_frames: self.overlap_frames,
            output_start,
        })
    }
}
