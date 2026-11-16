use super::*;
use anyhow::{ensure, Result};

/// Completed normalized encoder outputs in the same order as the admitted
/// encoded references. Geometry and packed input are derived from these exact
/// tensors; no independently maintained image/audio lists can reorder them.
#[derive(Clone)]
pub(crate) enum ConditionedReference {
    Image(ManagedHostTensor),
    Audio(ManagedHostTensor),
    Video {
        video: ManagedHostTensor,
        audio: Option<ManagedHostTensor>,
    },
}
impl ConditionedReference {
    pub(super) fn backing_bytes(&self) -> usize {
        match self {
            Self::Image(t) | Self::Audio(t) => t.bytes().backing_bytes(),
            Self::Video { video, audio } => {
                video.bytes().backing_bytes()
                    + audio.as_ref().map_or(0, |t| t.bytes().backing_bytes())
            }
        }
    }
    pub(crate) fn latents(&self) -> FastH3ReferenceLatents<'_> {
        match self {
            Self::Image(t) => FastH3ReferenceLatents::Image(t),
            Self::Audio(t) => FastH3ReferenceLatents::Audio(t),
            Self::Video { video, audio } => FastH3ReferenceLatents::Video {
                video,
                audio: audio.as_ref(),
            },
        }
    }
    pub(crate) fn geometry(&self) -> Result<FastH3Reference> {
        Ok(match self {
            Self::Video { video, audio } => {
                let dims: [u64; 5] = video.tensor().dimensions().try_into()?;
                ensure!(
                    dims[..2] == [1, 24] && dims[2..].iter().all(|&n| n > 0),
                    "video encoder returned a different latent geometry"
                );
                let audio_latents_per_channel = audio
                    .as_ref()
                    .map(|t| -> Result<usize> {
                        let d: [u64; 3] = t.tensor().dimensions().try_into()?;
                        ensure!(
                            d[..2] == [2, 32] && d[2] > 0,
                            "video soundtrack returned a different latent geometry"
                        );
                        Ok(usize::try_from(d[2])?)
                    })
                    .transpose()?;
                FastH3Reference::Video {
                    latent_frames: usize::try_from(dims[2])?,
                    latent_height: usize::try_from(dims[3])?,
                    latent_width: usize::try_from(dims[4])?,
                    audio_latents_per_channel,
                }
            }
            Self::Image(t) => {
                let dims: [u64; 5] = t.tensor().dimensions().try_into()?;
                ensure!(
                    dims[..3] == [1, 24, 1],
                    "image encoder returned a different latent geometry"
                );
                FastH3Reference::Image {
                    latent_height: usize::try_from(dims[3])?,
                    latent_width: usize::try_from(dims[4])?,
                }
            }
            Self::Audio(t) => {
                let dims: [u64; 3] = t.tensor().dimensions().try_into()?;
                ensure!(
                    dims[..2] == [2, 32] && dims[2] > 0,
                    "audio encoder returned a different latent geometry"
                );
                FastH3Reference::Audio {
                    latents_per_channel: usize::try_from(dims[2])?,
                }
            }
        })
    }
}
