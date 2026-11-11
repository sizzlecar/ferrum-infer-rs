use super::*;
use serde::Deserializer;
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct PH3ImageVaeDecoderExecutionConfig {
    architecture: PH3ImageVaeDecoderConfig,
    batch: u64,
    latent_height: u64,
    latent_width: u64,
    execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    architecture: PH3ImageVaeDecoderConfig,
    batch: u64,
    latent_height: u64,
    latent_width: u64,
    execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
}
impl<'de> Deserialize<'de> for PH3ImageVaeDecoderExecutionConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(
            w.architecture,
            w.batch,
            w.latent_height,
            w.latent_width,
            w.execution_preset,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl PH3ImageVaeDecoderExecutionConfig {
    pub fn new(
        architecture: PH3ImageVaeDecoderConfig,
        batch: u64,
        latent_height: u64,
        latent_width: u64,
        execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            batch,
            latent_height,
            latent_width,
            execution_preset,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn architecture(&self) -> &PH3ImageVaeDecoderConfig {
        &self.architecture
    }
    pub const fn batch(&self) -> u64 {
        self.batch
    }
    pub const fn latent_height(&self) -> u64 {
        self.latent_height
    }
    pub const fn latent_width(&self) -> u64 {
        self.latent_width
    }
    pub fn patches(&self) -> Result<u64, VNextError> {
        self.latent_height
            .checked_mul(self.latent_width)
            .ok_or_else(|| invalid("patch count overflows"))
    }
    pub fn sequence_length(&self) -> Result<u64, VNextError> {
        self.architecture
            .sequence_length(self.latent_height, self.latent_width)
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.batch
            .checked_mul(self.sequence_length()?)
            .ok_or_else(|| invalid("decoder row count overflows"))
    }
    pub(super) fn block(&self, index: u64) -> Result<PH3ImageVaeDecoderBlockConfig, VNextError> {
        PH3ImageVaeDecoderBlockConfig::new(
            self.architecture.clone(),
            index,
            self.batch,
            self.sequence_length()?,
            self.execution_preset,
        )
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([
            (
                ProgramValueId::new(PH3ImageVaeDecoderFamily::INPUT_ID)?,
                tensor(vec![
                    self.batch
                        .checked_mul(self.patches()?)
                        .ok_or_else(|| invalid("input rows overflow"))?,
                    self.architecture.latent_channels(),
                ]),
            ),
            (
                ProgramValueId::new(PH3ImageVaeDecoderFamily::COSINE_ID)?,
                tensor(vec![self.rows()?, self.architecture.rotary_dim()]),
            ),
            (
                ProgramValueId::new(PH3ImageVaeDecoderFamily::SINE_ID)?,
                tensor(vec![self.rows()?, self.architecture.rotary_dim()]),
            ),
        ]))
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.architecture.validate()?;
        self.block(0)?;
        for t in self.input_tensor_specs()?.values() {
            t.validate("VAE decoder input")?;
        }
        if self.batch == 0 || self.latent_height == 0 || self.latent_width == 0 {
            return Err(invalid(
                "positive batch and latent image dimensions required",
            ));
        }
        Ok(())
    }
}
pub(super) fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
