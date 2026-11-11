use super::*;

/// Separate static installations within one fixed source domain. Sharing an
/// archive does not by itself establish a compatible native weight snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageWeightOwnerKey {
    TextEncoder,
    TextConnector,
    VisionEncoder,
    VaeEncoder,
    Denoiser,
    Decoder,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStageAttachment {
    #[default]
    MayInitializeStored,
    /// Install/export the complete immutable snapshot even when sessions use
    /// cold retirement. Later subset plans must attach to this exact owner.
    InitializeShared,
    /// No source read or cold fallback is permitted when this snapshot is
    /// absent, or when the sealed physical attachment contract differs.
    RequireShared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageStageDeclaration {
    pub stage: ImageStage,
    pub role: ImageStagePlanRole,
    pub batch_size: usize,
    pub weight_owner: Option<ImageWeightOwnerKey>,
    pub attachment: ImageStageAttachment,
}
impl ImageStageDeclaration {
    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(
            self.batch_size > 0,
            "image stage batch size must be positive"
        );
        ensure!(
            self.attachment == ImageStageAttachment::MayInitializeStored
                || self.weight_owner.is_some(),
            "shared-only stage requires an explicit weight owner"
        );
        Ok(())
    }
}
