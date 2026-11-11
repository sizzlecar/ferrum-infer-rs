use super::*;
use anyhow::{ensure, Result};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
/// One borrowed original archive/canonical source, covering every phase.
pub(crate) struct PH3ImageDiTCheckpoint<'a> {
    config: &'a PH3ImageDiTConfig,
    source: &'a ComfyCanonicalWeightSource,
    stack: PH3ImageCurveStackCheckpoint<'a>,
}
impl<'a> PH3ImageDiTCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageDiTConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageDiTLimits,
    ) -> Result<Self> {
        program::compose(config, limits)?;
        PH3ImageTokenRefinerCheckpoint::new(&config.refiner, source, limits.refiner())?;
        let stack = PH3ImageCurveStackCheckpoint::new(&config.stack, source, limits.stack())?;
        PH3ImageFinalLayerCheckpoint::new(
            &config.final_layer,
            source.archive(),
            limits.final_layer(),
        )?;
        for (name, shape) in projection::entries(config)? {
            let t = source.archive().tensor(name)?;
            ensure!(
                t.element_type() == Some(ElementType::F32) && t.shape() == shape,
                "DiT patch projection {name} original F32 shape/dtype differs"
            );
            ensure!(
                t.bytes()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "DiT patch projection {name} must be finite"
            );
        }
        Ok(Self {
            config,
            source,
            stack,
        })
    }
    pub fn config(&self) -> &'a PH3ImageDiTConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
    pub fn managed_basis(
        &self,
        host: &HostMemoryAccount,
        times: &[f32],
        arithmetic: super::super::PH3ImageCurveLerpArithmetic,
    ) -> Result<ManagedHostTensor> {
        self.stack.managed_basis(host, times, arithmetic)
    }
}
impl WeightComponentSource for PH3ImageDiTCheckpoint<'_> {
    fn component<'source>(
        &'source self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component(c)
    }
}
