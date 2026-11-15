//! Ordered encoded media and explicit model/transport capability bounds.
//! A reference retains its original Host owner; its kind does not claim that
//! container decoding or model encoding has completed.
use super::request::invalid;
use crate::iterative_service::ServiceError;
use ferrum_runtime::ManagedHostBytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoReferenceKind {
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone)]
pub enum VideoReference {
    Image(ManagedHostBytes),
    Video(ManagedHostBytes),
    Audio(ManagedHostBytes),
}
impl VideoReference {
    pub fn from_encoded(kind: VideoReferenceKind, bytes: ManagedHostBytes) -> Self {
        match kind {
            VideoReferenceKind::Image => Self::Image(bytes),
            VideoReferenceKind::Video => Self::Video(bytes),
            VideoReferenceKind::Audio => Self::Audio(bytes),
        }
    }
    pub fn kind(&self) -> VideoReferenceKind {
        match self {
            Self::Image(_) => VideoReferenceKind::Image,
            Self::Video(_) => VideoReferenceKind::Video,
            Self::Audio(_) => VideoReferenceKind::Audio,
        }
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        match self {
            Self::Image(bytes) | Self::Video(bytes) | Self::Audio(bytes) => bytes,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoReferenceLimits {
    pub minimum_visual: usize,
    pub maximum_images: usize,
    pub maximum_videos: usize,
    pub maximum_audio: usize,
    pub maximum_total: usize,
}
impl VideoReferenceLimits {
    pub fn validate(self) -> Result<(), ServiceError> {
        if self.minimum_visual > self.maximum_total
            || self.maximum_images > self.maximum_total
            || self.maximum_videos > self.maximum_total
            || self.maximum_audio > self.maximum_total
            || self.minimum_visual > self.maximum_images.saturating_add(self.maximum_videos)
            || self
                .maximum_total
                .checked_mul(std::mem::size_of::<VideoReference>())
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(invalid("invalid ordered media reference limits"));
        }
        Ok(())
    }
    pub fn validate_kinds(
        self,
        kinds: impl IntoIterator<Item = VideoReferenceKind>,
    ) -> Result<(), ServiceError> {
        self.validate()?;
        let (mut images, mut videos, mut audio, mut total) = (0usize, 0usize, 0usize, 0usize);
        for kind in kinds {
            total = total
                .checked_add(1)
                .ok_or_else(|| invalid("reference count overflow"))?;
            match kind {
                VideoReferenceKind::Image => images += 1,
                VideoReferenceKind::Video => videos += 1,
                VideoReferenceKind::Audio => audio += 1,
            }
            if total > self.maximum_total
                || images > self.maximum_images
                || videos > self.maximum_videos
                || audio > self.maximum_audio
            {
                return Err(invalid(
                    "ordered media reference count exceeds declared capability",
                ));
            }
        }
        if images + videos < self.minimum_visual {
            return Err(invalid("request requires a visual reference"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_media_counts_require_visual_and_enforce_each_and_total_bounds() {
        use VideoReferenceKind::*;
        let limits = VideoReferenceLimits {
            minimum_visual: 1,
            maximum_images: 2,
            maximum_videos: 0,
            maximum_audio: 2,
            maximum_total: 3,
        };
        for accepted in [
            vec![Image],
            vec![Audio, Image],
            vec![Image, Audio, Audio],
            vec![Image, Image, Audio],
        ] {
            limits.validate_kinds(accepted).unwrap();
        }
        for rejected in [
            vec![],
            vec![Audio],
            vec![Image, Video],
            vec![Image, Image, Image],
            vec![Image, Audio, Audio, Audio],
            vec![Image, Image, Audio, Audio],
        ] {
            assert!(limits.validate_kinds(rejected).is_err());
        }
        assert!(VideoReferenceLimits {
            minimum_visual: 3,
            ..limits
        }
        .validate()
        .is_err());
        assert!(VideoReferenceLimits {
            maximum_total: usize::MAX,
            ..limits
        }
        .validate()
        .is_err());
    }
}
