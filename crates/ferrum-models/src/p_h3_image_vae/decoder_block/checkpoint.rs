use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::{
    ComfyCanonicalWeightSource, ComfyQuantizationMetadata, ComfyQuantizedMatrix,
};
use half::bf16;
/// Borrows the original archive. Original BF16 vectors are checked before
/// their exact static promotion through the existing dense source materializer.
/// Four canonical I8 matrices stay compressed, including their -128 values.
pub struct PH3ImageVaeDecoderBlockCheckpoint<'a> {
    config: &'a PH3ImageVaeDecoderBlockConfig,
    source: &'a ComfyCanonicalWeightSource,
}
impl<'a> PH3ImageVaeDecoderBlockCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageVaeDecoderBlockConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageVaeDecoderBlockLimits,
    ) -> Result<Self> {
        PH3ImageVaeDecoderBlockFamily::new(limits)?.check(config)?;
        let expected = ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: true,
            convrot_groupsize: 256,
        };
        for (module, input, output) in weights::projections(config)? {
            let external = weights::external(config, module);
            let matrix = ComfyQuantizedMatrix::from_archive(source.archive(), &external)?;
            ensure!(matrix.metadata()==&expected,"decoder projection {external} requires its declared canonical INT8 ConvRot metadata");
            ensure!(
                matrix.rows() as u64 == output && matrix.columns() as u64 == input,
                "decoder projection {external} shape differs from architecture"
            );
        }
        for (name, shape) in weights::original_bf16_vectors(config)? {
            let external = weights::external(config, &name);
            let t = source.archive().tensor(&external)?;
            ensure!(
                t.dtype() == safetensors::Dtype::BF16 && t.shape() == shape,
                "decoder parameter {external} must retain its original BF16 shape"
            );
            ensure!(
                t.bytes()
                    .chunks_exact(2)
                    .all(
                        |b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()
                    ),
                "decoder parameter {external} contains nonfinite values"
            );
        }
        Ok(Self { config, source })
    }
    pub fn config(&self) -> &'a PH3ImageVaeDecoderBlockConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
}
impl WeightComponentSource for PH3ImageVaeDecoderBlockCheckpoint<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component(component)
    }
}
