//! Original causal encoder clip boundaries: repeat the final source frame in
//! the last clip, concatenate posterior moments, then drop the suffix once.
use super::FastH3VideoVaeConfig;
use anyhow::{ensure, Context, Result};

#[derive(Clone, Debug)]
pub struct FastH3VideoEncodePlan {
    input_frames: usize,
    clip_frames: usize,
    clip_moments: usize,
    clips: usize,
    output_moments: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastH3VideoEncodeChunk {
    pub source_start: usize,
    pub source_frames: usize,
    pub repeat_last_frames: usize,
    pub output_start: usize,
    pub publish_moments: usize,
}
impl FastH3VideoEncodePlan {
    pub fn new(
        config: &FastH3VideoVaeConfig,
        input_frames: usize,
        maximum_output_moments: usize,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(input_frames > 0, "video encoder requires nonempty frames");
        let clip_frames = config.clip_length();
        let clip_moments = clip_frames.div_ceil(config.temporal_compression_ratio()?);
        let clips = input_frames.div_ceil(clip_frames);
        let output_moments = clips
            .checked_mul(clip_moments)
            .and_then(|v| v.checked_sub(config.token_drop()))
            .context("video encoder temporal extent overflows")?;
        ensure!(
            output_moments > 0 && output_moments <= maximum_output_moments,
            "video encoder moments exceed declared capacity"
        );
        Ok(Self {
            input_frames,
            clip_frames,
            clip_moments,
            clips,
            output_moments,
        })
    }
    pub fn chunk_count(&self) -> usize {
        self.clips
    }
    pub fn clip_frames(&self) -> usize {
        self.clip_frames
    }
    pub fn clip_moments(&self) -> usize {
        self.clip_moments
    }
    pub fn output_moments(&self) -> usize {
        self.output_moments
    }
    pub fn chunk(&self, index: usize) -> Result<FastH3VideoEncodeChunk> {
        ensure!(
            index < self.clips,
            "video encoder chunk outside declared input"
        );
        let source_start = index
            .checked_mul(self.clip_frames)
            .context("clip source offset overflow")?;
        let source_frames = (self.input_frames - source_start).min(self.clip_frames);
        let output_start = index * self.clip_moments;
        Ok(FastH3VideoEncodeChunk {
            source_start,
            source_frames,
            repeat_last_frames: self.clip_frames - source_frames,
            output_start,
            publish_moments: (self.output_moments - output_start).min(self.clip_moments),
        })
    }
    pub fn source_frame(&self, chunk: usize, frame: usize) -> Result<usize> {
        ensure!(
            frame < self.clip_frames,
            "frame outside original encoder clip"
        );
        let c = self.chunk(chunk)?;
        Ok(c.source_start + frame.min(c.source_frames - 1))
    }
    /// FastVideo trims a reference to complete 17n+5 input chunks. A slice
    /// shorter than the first complete boundary remains intact and is padded
    /// by the clip plan above. Constants are derived from the declared VAE.
    pub fn reference_frames(config: &FastH3VideoVaeConfig, decoded_frames: usize) -> Result<usize> {
        config.validate()?;
        ensure!(decoded_frames > 0, "reference video requires a frame");
        let clip = config.clip_length();
        let moments = clip.div_ceil(config.temporal_compression_ratio()?);
        let complete = (decoded_frames.saturating_sub(moments) / clip).max(1);
        let boundary = complete
            .checked_mul(clip)
            .and_then(|v| v.checked_add(moments))
            .context("reference trim extent overflows")?;
        Ok(decoded_frames.min(boundary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_clip_padding_concat_and_single_suffix_drop_cover_temporal_boundaries() {
        let c = super::super::tests::config();
        for frames in 1..200 {
            let p = FastH3VideoEncodePlan::new(&c, frames, 1000).unwrap();
            // Independently enumerate the original repeat-last + clip loop.
            let mut padded: Vec<_> = (0..frames).collect();
            while padded.len() % 17 != 0 {
                padded.push(frames - 1);
            }
            let mut starts = vec![];
            for clip in padded.chunks(17) {
                starts.extend([clip[0], clip[4], clip[8], clip[12], clip[16]]);
            }
            starts.truncate(starts.len() - 3);
            let mut observed = vec![];
            for i in 0..p.chunk_count() {
                let chunk = p.chunk(i).unwrap();
                assert_eq!(chunk.source_frames + chunk.repeat_last_frames, 17);
                assert_eq!(chunk.output_start, observed.len());
                for j in 0..17 {
                    assert_eq!(p.source_frame(i, j).unwrap(), padded[i * 17 + j]);
                }
                for j in 0..chunk.publish_moments {
                    observed.push(p.source_frame(i, j * 4).unwrap());
                }
            }
            assert_eq!(observed, starts);
            assert_eq!(p.output_moments(), starts.len());
            assert!(p.chunk(p.chunk_count()).is_err());
            assert!(p.source_frame(0, 17).is_err());
            assert!(FastH3VideoEncodePlan::new(&c, frames, starts.len() - 1).is_err());
        }
        for (n, trimmed) in [
            (1, 1),
            (5, 5),
            (21, 21),
            (22, 22),
            (23, 22),
            (38, 22),
            (39, 39),
            (40, 39),
            (56, 56),
        ] {
            assert_eq!(
                FastH3VideoEncodePlan::reference_frames(&c, n).unwrap(),
                trimmed
            );
        }
        assert!(FastH3VideoEncodePlan::new(&c, 0, 1000).is_err());
        assert!(FastH3VideoEncodePlan::new(&c, usize::MAX, 1000).is_err());
        assert!(FastH3VideoEncodePlan::reference_frames(&c, 0).is_err());
    }
}
