use super::*;
use anyhow::Result;
use ferrum_quantization::SafetensorsArchive;
pub(crate) struct PH3ImageVaeEncoderCheckpoint<'a> {
    config: &'a PH3ImageVaeEncoderConfig,
    source: &'a SafetensorsArchive,
    host: HostMemoryAccount,
}
impl<'a> PH3ImageVaeEncoderCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageVaeEncoderConfig,
        source: &'a SafetensorsArchive,
        host: HostMemoryAccount,
        limits: PH3ImageVaeEncoderLimits,
    ) -> Result<Self> {
        super::checkpoint_source::validate(config, source, limits)?;
        Ok(Self {
            config,
            source,
            host,
        })
    }
    /// Internal projection of an immutable adapter source already validated
    /// once and checked against the prepared physical Stored schema. This is
    /// not an alternative public source-validation entrypoint.
    pub(in crate::p_h3_image_vae) fn view_validated(
        config: &'a PH3ImageVaeEncoderConfig,
        source: &'a SafetensorsArchive,
        host: HostMemoryAccount,
    ) -> Self {
        Self {
            config,
            source,
            host,
        }
    }
    pub fn config(&self) -> &'a PH3ImageVaeEncoderConfig {
        self.config
    }
    pub fn source(&self) -> &'a SafetensorsArchive {
        self.source
    }
}
impl WeightComponentSource for PH3ImageVaeEncoderCheckpoint<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        super::checkpoint_source::component(self.config, self.source, &self.host, c)
    }
}
