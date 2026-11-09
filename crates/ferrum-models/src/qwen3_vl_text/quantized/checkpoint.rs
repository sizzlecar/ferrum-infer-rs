use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizedMatrix};
use half::bf16;

/// Borrows original quantized storage and promotes only small BF16 norm
/// vectors through the existing dense source materializer. It never expands
/// a quantized projection. Construction verifies this exact configuration's
/// metadata, component shapes, raw scale dtypes and original norm dtypes.
pub struct Qwen3VLTextComfyCheckpoint<'a> {
    config: &'a Qwen3VLTextComfyConfig,
    source: &'a ComfyCanonicalWeightSource,
}
impl<'a> Qwen3VLTextComfyCheckpoint<'a> {
    pub fn new(
        config: &'a Qwen3VLTextComfyConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: Qwen3VLTextComfyLimits,
    ) -> Result<Self> {
        Qwen3VLTextComfyFamily::new(limits)?.check(config)?;
        for (module, input, output) in weights::projection_shapes(config.text())? {
            let matrix = ComfyQuantizedMatrix::from_archive(source.archive(), &module)?;
            ensure!(
                matrix.metadata() == &config.projections()[&module].metadata(),
                "projection {module} metadata differs from the compiled execution route"
            );
            ensure!(
                matrix.columns() as u64 == input && matrix.rows() as u64 == output,
                "projection {module} logical shape differs from the text architecture"
            );
        }
        for (name, dimensions) in weights::norm_shapes(config.text()) {
            let tensor = source.archive().tensor(&name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::BF16 && tensor.shape() == dimensions,
                "norm {name} must retain its original BF16 checkpoint shape"
            );
            ensure!(
                tensor
                    .bytes()
                    .chunks_exact(2)
                    .all(
                        |b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()
                    ),
                "norm {name} contains a nonfinite BF16 value"
            );
        }
        Ok(Self { config, source })
    }
    pub fn config(&self) -> &'a Qwen3VLTextComfyConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
}
impl WeightComponentSource for Qwen3VLTextComfyCheckpoint<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component(component)
    }
}
