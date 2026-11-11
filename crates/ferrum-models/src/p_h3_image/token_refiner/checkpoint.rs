//! Immutable source provenance, preserving raw INT8 matrices and F32 scales.
use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::{
    ComfyCanonicalWeightSource, ComfyQuantizationMetadata, ComfyQuantizedMatrix,
};
use half::bf16;
pub struct PH3ImageTokenRefinerCheckpoint<'a> {
    config: &'a PH3ImageTokenRefinerConfig,
    source: &'a ComfyCanonicalWeightSource,
}
impl<'a> PH3ImageTokenRefinerCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageTokenRefinerConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageTokenRefinerLimits,
    ) -> Result<Self> {
        PH3ImageTokenRefinerFamily::new(limits)?.check(config)?;
        for (name, shape, dtype) in weights::entries(config) {
            let tensor = source.archive().tensor(&name)?;
            ensure!(
                tensor.shape() == shape && tensor.element_type() == Some(dtype),
                "token refiner checkpoint {name} shape/dtype differs"
            );
            match dtype {
                ElementType::Bf16 => {
                    ensure!(
                        tensor.bytes().chunks_exact(2).all(|b| bf16::from_bits(
                            u16::from_le_bytes(b.try_into().unwrap())
                        )
                        .is_finite()),
                        "token refiner BF16 parameters must be finite"
                    )
                }
                ElementType::F32 => ensure!(
                    tensor
                        .bytes()
                        .chunks_exact(4)
                        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                    "token refiner scales must be finite"
                ),
                _ => {}
            }
        }
        let expected = ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: true,
            convrot_groupsize: 256,
        };
        for layer in 0..config.layers {
            for name in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
                let matrix = ComfyQuantizedMatrix::from_archive(
                    source.archive(),
                    &format!("token_refiner.blocks.{layer}.{name}"),
                )?;
                ensure!(
                    matrix.metadata() == &expected,
                    "token refiner projection is not the declared canonical W8 H256 route"
                );
            }
        }
        Ok(Self { config, source })
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
    pub fn config(&self) -> &'a PH3ImageTokenRefinerConfig {
        self.config
    }
}
