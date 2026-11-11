//! Ordered reference-image layout. The VLM image slots remain explicit in
//! TEXT input and are replaced by four-times as many VAE rows in the DiT joint
//! sequence. They are never mistaken for an ordinary text prefix.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenImage21ReferenceImage {
    image_slot_start: u32,
    image_slots: u32,
    latent_height: u32,
    latent_width: u32,
}
impl QwenImage21ReferenceImage {
    /// Slot coordinates address the original, cropped VLM sequence. Layout
    /// construction additionally checks order, valid prompt lengths and RoPE.
    pub fn new(
        image_slot_start: u32,
        image_slots: u32,
        latent_height: u32,
        latent_width: u32,
    ) -> Result<Self, VNextError> {
        let reference = Self {
            image_slot_start,
            image_slots,
            latent_height,
            latent_width,
        };
        reference.validate()?;
        Ok(reference)
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.latent_height == 0
            || self.latent_width == 0
            || self.latent_height > 2048
            || self.latent_width > 2048
            || self.latent_height % 2 != 0
            || self.latent_width % 2 != 0
            || self.image_slot_start == 0
            || self.image_slots == 0
            || self.latent_tokens() != u64::from(self.image_slots) * 4
        {
            return Err(invalid("reference_image","requires positive even latent axes <=2048, a text prefix, and exactly four latent rows per VLM image slot"));
        }
        Ok(())
    }
    pub const fn image_slot_start(&self) -> u32 {
        self.image_slot_start
    }
    pub const fn image_slots(&self) -> u32 {
        self.image_slots
    }
    pub const fn latent_height(&self) -> u32 {
        self.latent_height
    }
    pub const fn latent_width(&self) -> u32 {
        self.latent_width
    }
    pub fn latent_tokens(&self) -> u64 {
        u64::from(self.latent_height) * u64::from(self.latent_width)
    }
}
impl QwenImage21T2iLayout {
    /// Compatibility accessor: multi-image layouts are not single-image edits.
    pub fn reference_image(&self) -> Option<&QwenImage21ReferenceImage> {
        match self.reference_images.as_slice() {
            [reference] => Some(reference),
            _ => None,
        }
    }
    pub fn reference_images(&self) -> &[QwenImage21ReferenceImage] {
        &self.reference_images
    }
    /// Per-sample input rows, concatenated in reference-image order.
    pub fn reference_latent_tokens(&self) -> u64 {
        self.reference_images
            .iter()
            .map(|r| r.latent_tokens())
            .sum()
    }
    pub fn target_start(&self) -> u64 {
        self.sequence_length() - self.target_tokens()
    }
    pub fn with_reference_images(
        mut self,
        references: Vec<QwenImage21ReferenceImage>,
    ) -> Result<Self, VNextError> {
        if references.len() > 10 {
            return Err(invalid(
                "reference_images",
                "at most ten reference images are supported",
            ));
        }
        let mut after = 0;
        let mut position = 0_u64;
        for r in &references {
            r.validate()?;
            if r.image_slot_start < after {
                return Err(invalid(
                    "reference_images",
                    "image slot intervals must be ordered and nonoverlapping",
                ));
            }
            position += u64::from(r.image_slot_start - after)
                + u64::from(r.latent_height.max(r.latent_width));
            after = r
                .image_slot_start
                .checked_add(r.image_slots)
                .ok_or_else(|| invalid("reference_images", "image slot interval overflows"))?;
            if after >= self.text_tokens || self.valid_text_lengths.iter().any(|&n| n < after) {
                return Err(invalid(
                    "reference_images",
                    "image slots must fit every valid VLM prefix and leave a suffix text segment",
                ));
            }
        }
        let target_frame = position + u64::from(self.text_tokens - after);
        if target_frame >= 8192 {
            return Err(invalid(
                "reference_images",
                "interleaved rotary frame exceeds positive frequency domain",
            ));
        }
        self.reference_images = references;
        u32::try_from(self.sequence_length())
            .map_err(|_| invalid("reference_image", "joint sequence overflows u32"))?;
        self.attention_layout()?;
        Ok(self)
    }
    pub(super) fn edit_rotary_position(&self, row: u64) -> Option<[i32; 3]> {
        if row >= self.sequence_length() {
            return None;
        }
        let mut text_cursor = 0;
        let mut joint_cursor = 0_u64;
        let mut position = 0_u64;
        for r in &self.reference_images {
            let text_length = u64::from(r.image_slot_start - text_cursor);
            if row < joint_cursor + text_length {
                return Some([(position + row - joint_cursor) as i32; 3]);
            }
            joint_cursor += text_length;
            position += text_length;
            if row < joint_cursor + r.latent_tokens() {
                return Some(centered(
                    row - joint_cursor,
                    position,
                    r.latent_height,
                    r.latent_width,
                ));
            }
            joint_cursor += r.latent_tokens();
            position += u64::from(r.latent_height.max(r.latent_width));
            text_cursor = r.image_slot_start + r.image_slots;
        }
        if row < self.target_start() {
            return Some([(position + row - joint_cursor) as i32; 3]);
        }
        let target_frame = position + u64::from(self.text_tokens - text_cursor);
        Some(centered(
            row - self.target_start(),
            target_frame,
            self.latent_height,
            self.latent_width,
        ))
    }
    pub(super) fn edit_attention_layout(&self) -> Result<QwenImage21AttentionLayout, VNextError> {
        let refs = u32::try_from(self.reference_latent_tokens())
            .map_err(|_| invalid("reference_image", "reference extent overflows"))?;
        let mut slots = 0;
        let mut text_cursor = 0;
        let mut segments = Vec::new();
        for r in &self.reference_images {
            if r.image_slot_start > text_cursor {
                segments.push(QwenImage21AttentionSegment::new(
                    r.image_slot_start - text_cursor,
                    false,
                ));
            }
            // Adjacent images are distinct bidirectional blocks, not one run.
            segments.push(QwenImage21AttentionSegment::new(
                r.latent_tokens() as u32,
                true,
            ));
            text_cursor = r.image_slot_start + r.image_slots;
            slots += r.image_slots;
        }
        let target = self.target_tokens() as u32;
        segments.extend([
            QwenImage21AttentionSegment::new(self.text_tokens - text_cursor, false),
            QwenImage21AttentionSegment::new(target, true),
        ]);
        let key_runs = self
            .valid_text_lengths
            .iter()
            .map(|&valid| {
                let prefix_valid = valid - slots + refs;
                let padding = self.text_tokens - valid;
                if padding == 0 {
                    vec![QwenImage21AttentionKeyRun::new(prefix_valid + target, true)]
                } else {
                    vec![
                        QwenImage21AttentionKeyRun::new(prefix_valid, true),
                        QwenImage21AttentionKeyRun::new(padding, false),
                        QwenImage21AttentionKeyRun::new(target, true),
                    ]
                }
            })
            .collect();
        QwenImage21AttentionLayout::new(
            u32::try_from(self.sequence_length())
                .map_err(|_| invalid("reference_image", "joint sequence overflows"))?,
            segments,
            key_runs,
        )
    }
}
fn centered(index: u64, frame: u64, height: u32, width: u32) -> [i32; 3] {
    [
        frame as i32,
        (index / u64::from(width)) as i32 - (height - height / 2) as i32,
        (index % u64::from(width)) as i32 - (width - width / 2) as i32,
    ]
}
impl QwenImage21TransformerConfig {
    /// Image slots are in the cropped VLM sequence, before the DiT fourfold
    /// replacement. Target latents remain a separate input and output domain.
    pub fn with_reference_image(
        self,
        image_slot_start: u32,
        image_slots: u32,
        latent_height: u32,
        latent_width: u32,
    ) -> Result<Self, VNextError> {
        self.with_reference_images(vec![QwenImage21ReferenceImage::new(
            image_slot_start,
            image_slots,
            latent_height,
            latent_width,
        )?])
    }
    /// Reference latents use one sample-major tensor: each sample contains all
    /// its images in this order. Only the target rows are returned as velocity.
    pub fn with_reference_images(
        mut self,
        references: Vec<QwenImage21ReferenceImage>,
    ) -> Result<Self, VNextError> {
        self.layout = self.layout.with_reference_images(references)?;
        self.validate()?;
        Ok(self)
    }
}
