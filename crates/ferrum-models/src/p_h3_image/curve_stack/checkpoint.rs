//! One borrowed canonical source across all layers and the shared time table.
use super::super::{PH3ImageCurveBlockCheckpoint, PH3ImageCurveLerpArithmetic};
use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizedMatrix};
use half::bf16;
pub struct PH3ImageCurveStackCheckpoint<'a> {
    config: &'a PH3ImageCurveStackConfig,
    base: PH3ImageCurveBlockCheckpoint<'a>,
}
impl<'a> PH3ImageCurveStackCheckpoint<'a> {
    pub fn new(
        c: &'a PH3ImageCurveStackConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageCurveStackLimits,
    ) -> Result<Self> {
        let complete = compose::build(c, limits)?;
        let base = PH3ImageCurveBlockCheckpoint::new(&c.block, source, limits.block())?;
        for component in &complete.schema.components {
            // The generic archive source may transcode dense parameters. This
            // checkpoint profile instead promises the original physical dtype
            // at every layer, so inspect its header before requesting a port.
            let [name] = component.external_names.as_slice() else {
                anyhow::bail!("stack parameter requires one original checkpoint tensor");
            };
            let original = source.archive().tensor(name)?;
            let dtype_matches = if name.ends_with(".weight_s_rel") {
                component.physical_element_type() == ElementType::U8
                    && original.dtype() == safetensors::Dtype::F8_E4M3
            } else {
                original.element_type() == Some(component.physical_element_type())
            };
            ensure!(
                original.shape() == component.dimensions && dtype_matches,
                "stack checkpoint {name} original shape/dtype differs"
            );
            let payload = source.component(component)?;
            match payload.element_type() {
                ElementType::Bf16 => {
                    ensure!(
                        payload.bytes().chunks_exact(2).all(|b| bf16::from_bits(
                            u16::from_le_bytes(b.try_into().unwrap())
                        )
                        .is_finite()),
                        "stack BF16 parameter must be finite"
                    )
                }
                ElementType::F32 => ensure!(
                    payload
                        .bytes()
                        .chunks_exact(4)
                        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                    "stack F32 parameter must be finite"
                ),
                _ => {}
            }
        }
        for layer in 0..c.layers {
            for (module, expected) in c.storage_for_layer(layer)?.modules() {
                let matrix = ComfyQuantizedMatrix::from_archive(
                    source.archive(),
                    &format!("blocks.{layer}.{module}"),
                )?;
                ensure!(
                    PH3ImageProjectionStorage::from_metadata(matrix.metadata())? == expected,
                    "stack projection differs from its declared canonical storage"
                );
            }
        }
        Ok(Self { config: c, base })
    }
    pub fn config(&self) -> &'a PH3ImageCurveStackConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.base.source()
    }
    pub fn managed_basis(
        &self,
        host: &HostMemoryAccount,
        times: &[f32],
        arithmetic: PH3ImageCurveLerpArithmetic,
    ) -> Result<ManagedHostTensor> {
        self.base.managed_basis(host, times, arithmetic)
    }
}
