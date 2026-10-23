//! Single-reference editing layout. The VLM image slots remain explicit in
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
    pub fn reference_image(&self) -> Option<&QwenImage21ReferenceImage> {
        self.reference_image.as_ref()
    }
    pub fn target_start(&self) -> u64 {
        self.sequence_length() - self.target_tokens()
    }
    pub(super) fn with_reference(
        mut self,
        r: QwenImage21ReferenceImage,
    ) -> Result<Self, VNextError> {
        if r.latent_height == 0
            || r.latent_width == 0
            || r.latent_height > 2048
            || r.latent_width > 2048
            || r.latent_height % 2 != 0
            || r.latent_width % 2 != 0
            || r.image_slot_start == 0
            || r.image_slots == 0
            || r.latent_tokens() != u64::from(r.image_slots) * 4
        {
            return Err(invalid("reference_image","requires positive even latent axes <=2048, a text prefix, and exactly four latent rows per VLM image slot"));
        }
        let after = r
            .image_slot_start
            .checked_add(r.image_slots)
            .ok_or_else(|| invalid("reference_image", "image slot interval overflows"))?;
        if after >= self.text_tokens || self.valid_text_lengths.iter().any(|&n| n < after) {
            return Err(invalid(
                "reference_image",
                "image slots must fit every valid VLM prefix and leave a suffix text segment",
            ));
        }
        let target_frame = u64::from(r.image_slot_start)
            + u64::from(r.latent_height.max(r.latent_width))
            + u64::from(self.text_tokens - after);
        if target_frame >= 8192 {
            return Err(invalid(
                "reference_image",
                "interleaved rotary frame exceeds positive frequency domain",
            ));
        }
        self.reference_image = Some(r);
        u32::try_from(self.sequence_length())
            .map_err(|_| invalid("reference_image", "joint sequence overflows u32"))?;
        self.attention_layout()?;
        Ok(self)
    }
    pub(super) fn edit_rotary_position(&self, row: u64) -> Option<[i32; 3]> {
        if row >= self.sequence_length() {
            return None;
        }
        let r = self.reference_image.as_ref()?;
        let t0 = u64::from(r.image_slot_start);
        let image_end = t0 + r.latent_tokens();
        if row < t0 {
            return Some([row as i32; 3]);
        }
        if row < image_end {
            return Some(centered(row - t0, t0, r.latent_height, r.latent_width));
        }
        let suffix_frame = t0 + u64::from(r.latent_height.max(r.latent_width));
        if row < self.target_start() {
            return Some([(suffix_frame + row - image_end) as i32; 3]);
        }
        let target_frame = suffix_frame + self.target_start() - image_end;
        Some(centered(
            row - self.target_start(),
            target_frame,
            self.latent_height,
            self.latent_width,
        ))
    }
    pub(super) fn edit_attention_layout(&self) -> Result<QwenImage21AttentionLayout, VNextError> {
        let r = self
            .reference_image
            .as_ref()
            .ok_or_else(|| invalid("reference_image", "missing checked reference"))?;
        let t0 = r.image_slot_start;
        let refs = u32::try_from(r.latent_tokens())
            .map_err(|_| invalid("reference_image", "reference extent overflows"))?;
        let suffix = self.text_tokens - t0 - r.image_slots;
        let target = self.target_tokens() as u32;
        let segments = vec![
            QwenImage21AttentionSegment::new(t0, false),
            QwenImage21AttentionSegment::new(refs, true),
            QwenImage21AttentionSegment::new(suffix, false),
            QwenImage21AttentionSegment::new(target, true),
        ];
        let key_runs = self
            .valid_text_lengths
            .iter()
            .map(|&valid| {
                let prefix_valid = valid - r.image_slots + refs;
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
        mut self,
        image_slot_start: u32,
        image_slots: u32,
        latent_height: u32,
        latent_width: u32,
    ) -> Result<Self, VNextError> {
        self.layout = self.layout.with_reference(QwenImage21ReferenceImage {
            image_slot_start,
            image_slots,
            latent_height,
            latent_width,
        })?;
        self.validate()?;
        Ok(self)
    }
}
