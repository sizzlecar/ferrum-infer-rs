use super::*;
use serde::Deserializer;
/// Full source CNN architecture. The vertical selects actual module indices
/// from this inventory; filenames and model-name substrings grant no behavior.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PH3ImageVaeEncoderArchitecture {
    in_channels: u64,
    latent_channels: u64,
    block_out_channels: Vec<u64>,
    layers_per_block: u64,
    spatial_downsample_factors: Vec<u64>,
    temporal_downsample_factors: Vec<u64>,
    norm_num_groups: u64,
    norm_epsilon: CanonicalRational,
}
#[derive(Deserialize)]
struct Hf {
    #[serde(rename = "_class_name")]
    class: String,
    in_channels: u64,
    latent_channels: u64,
    block_out_channels: Vec<u64>,
    layers_per_block: u64,
    spatial_downsample_factors: Vec<u64>,
    temporal_downsample_factors: Vec<u64>,
    norm_num_groups: u64,
    norm_eps: serde_json::Number,
    spatial_padding_mode: String,
}
impl PH3ImageVaeEncoderArchitecture {
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.is_empty() || bytes.len() > 4 << 20 {
            return Err(invalid("bounded nonempty metadata required"));
        }
        let h: Hf = serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
        if h.class != crate::p_h3_image_vae::PH3_IMAGE_VAE_CLASS
            || h.spatial_padding_mode != "reflect"
        {
            return Err(invalid(
                "requires declared image VAE class and literal reflect spatial semantics",
            ));
        }
        let a = Self {
            in_channels: h.in_channels,
            latent_channels: h.latent_channels,
            block_out_channels: h.block_out_channels,
            layers_per_block: h.layers_per_block,
            spatial_downsample_factors: h.spatial_downsample_factors,
            temporal_downsample_factors: h.temporal_downsample_factors,
            norm_num_groups: h.norm_num_groups,
            norm_epsilon: CanonicalRational::from_decimal_str(&h.norm_eps.to_string())?,
        };
        a.validate()?;
        Ok(a)
    }
    pub fn channels(&self) -> &[u64] {
        &self.block_out_channels
    }
    pub const fn in_channels(&self) -> u64 {
        self.in_channels
    }
    pub const fn latent_channels(&self) -> u64 {
        self.latent_channels
    }
    pub const fn layers_per_block(&self) -> u64 {
        self.layers_per_block
    }
    pub fn spatial_downsample_factors(&self) -> &[u64] {
        &self.spatial_downsample_factors
    }
    pub fn temporal_downsample_factors(&self) -> &[u64] {
        &self.temporal_downsample_factors
    }
    pub const fn norm_groups(&self) -> u64 {
        self.norm_num_groups
    }
    pub const fn epsilon(&self) -> CanonicalRational {
        self.norm_epsilon
    }
    pub(crate) fn validate(&self) -> Result<(), VNextError> {
        let n = self.block_out_channels.len();
        if self.in_channels == 0
            || self.latent_channels == 0
            || self.layers_per_block == 0
            || self.norm_num_groups == 0
            || n < 2
            || n > 128
            || self.spatial_downsample_factors.len() != n
            || self.temporal_downsample_factors.len() != n
            || self
                .block_out_channels
                .iter()
                .any(|&c| c == 0 || !c.is_multiple_of(self.norm_num_groups))
            || self
                .spatial_downsample_factors
                .iter()
                .chain(&self.temporal_downsample_factors)
                .any(|v| !matches!(v, 1 | 2))
        {
            return Err(invalid("CNN requires bounded positive levels, group-divisible widths and declared per-level strides 1 or 2"));
        }
        if self.norm_epsilon.numerator() <= 0
            || self.norm_epsilon.numerator() as i128 * 1_000_000_000_000
                < i128::from(self.norm_epsilon.denominator())
            || self.norm_epsilon.numerator() as u64 > self.norm_epsilon.denominator()
        {
            return Err(invalid("epsilon must be in [1e-12,1]"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct PH3ImageVaeEncoderVerticalConfig {
    architecture: PH3ImageVaeEncoderArchitecture,
    batch: u64,
    height: u64,
    width: u64,
    block_level: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    architecture: PH3ImageVaeEncoderArchitecture,
    batch: u64,
    height: u64,
    width: u64,
    block_level: u64,
}
impl<'de> Deserialize<'de> for PH3ImageVaeEncoderVerticalConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(w.architecture, w.batch, w.height, w.width, w.block_level)
            .map_err(serde::de::Error::custom)
    }
}
impl PH3ImageVaeEncoderVerticalConfig {
    pub fn new(
        architecture: PH3ImageVaeEncoderArchitecture,
        batch: u64,
        height: u64,
        width: u64,
        block_level: u64,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            batch,
            height,
            width,
            block_level,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn architecture(&self) -> &PH3ImageVaeEncoderArchitecture {
        &self.architecture
    }
    pub const fn batch(&self) -> u64 {
        self.batch
    }
    pub const fn height(&self) -> u64 {
        self.height
    }
    pub const fn width(&self) -> u64 {
        self.width
    }
    pub const fn block_level(&self) -> u64 {
        self.block_level
    }
    pub(super) fn widths(&self) -> (u64, u64) {
        (
            self.architecture.block_out_channels[0],
            self.architecture.block_out_channels[self.block_level as usize],
        )
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.architecture.validate()?;
        let a = &self.architecture;
        let level = usize::try_from(self.block_level)
            .map_err(|_| invalid("level exceeds address space"))?;
        if level == 0
            || level >= a.channels().len()
            || a.channels()[level - 1] != a.channels()[0]
            || a.channels()[level] == a.channels()[level - 1]
            || a.spatial_downsample_factors[level] != 2
            || self.batch == 0
            || self.height < 2
            || self.width < 2
        {
            return Err(invalid("vertical requires a declared width-changing level accepting stem width, spatial stride2, positive batch and reflect-valid dimensions"));
        }
        tensor(vec![self.batch, a.in_channels, self.height, self.width])
            .validate("encoder input")?;
        Ok(())
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(PH3ImageVaeEncoderVerticalFamily::INPUT_ID)?,
            tensor(vec![
                self.batch,
                self.architecture.in_channels,
                self.height,
                self.width,
            ]),
        )]))
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        program::compile_options(self)
    }
}
pub(super) fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
