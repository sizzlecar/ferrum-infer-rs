use super::*;
use anyhow::Result;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
/// All source tensors share one immutable archive. Small original BF16 values
/// promote into the actual caller's WeightBacking account; raw INT8 matrices
/// and F32 scales retain their archive owner and are never expanded on Host.
pub(crate) struct PH3ImageVaeDecoderCheckpoint<'a> {
    config: &'a PH3ImageVaeDecoderExecutionConfig,
    source: &'a ComfyCanonicalWeightSource,
    host: HostMemoryAccount,
}
impl<'a> PH3ImageVaeDecoderCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageVaeDecoderExecutionConfig,
        source: &'a ComfyCanonicalWeightSource,
        host: HostMemoryAccount,
        limits: PH3ImageVaeDecoderLimits,
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
        config: &'a PH3ImageVaeDecoderExecutionConfig,
        source: &'a ComfyCanonicalWeightSource,
        host: HostMemoryAccount,
    ) -> Self {
        Self {
            config,
            source,
            host,
        }
    }
    pub fn config(&self) -> &'a PH3ImageVaeDecoderExecutionConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
}
impl WeightComponentSource for PH3ImageVaeDecoderCheckpoint<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        super::checkpoint_source::component(self.config, self.source, &self.host, c)
    }
}
