use super::*;
use serde::Deserializer;
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct PH3ImageVaeEncoderConfig {
    architecture: PH3ImageVaeEncoderArchitecture,
    batch: u64,
    height: u64,
    width: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    architecture: PH3ImageVaeEncoderArchitecture,
    batch: u64,
    height: u64,
    width: u64,
}
impl<'de> Deserialize<'de> for PH3ImageVaeEncoderConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(w.architecture, w.batch, w.height, w.width).map_err(serde::de::Error::custom)
    }
}
impl PH3ImageVaeEncoderConfig {
    pub fn new(
        architecture: PH3ImageVaeEncoderArchitecture,
        batch: u64,
        height: u64,
        width: u64,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            batch,
            height,
            width,
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
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(PH3ImageVaeEncoderFamily::INPUT_ID)?,
            tensor(vec![
                self.batch,
                self.architecture.in_channels(),
                self.height,
                self.width,
            ]),
        )]))
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        program::compile_options(self)
    }
    pub fn output_tensor_spec(&self) -> Result<ProgramTensorSpec, VNextError> {
        let options = self.compile_options()?;
        Ok(
            options.tensor_specs()[&ProgramValueId::new(PH3ImageVaeEncoderFamily::OUTPUT_ID)?]
                .clone(),
        )
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.architecture.validate()?;
        if self.batch == 0 || self.height < 2 || self.width < 2 {
            return Err(invalid(
                "positive batch and reflect-valid image dimensions required",
            ));
        }
        let inventory = self
            .architecture
            .channels()
            .len()
            .checked_mul(
                usize::try_from(self.architecture.layers_per_block())
                    .map_err(|_| invalid("block count exceeds address space"))?,
            )
            .and_then(|n| n.checked_mul(10))
            .and_then(|n| n.checked_add(self.architecture.channels().len() * 2 + 8))
            .filter(|n| *n <= 65536)
            .ok_or_else(|| invalid("complete source inventory exceeds 65536 tensors"))?;
        let _ = inventory;
        for t in self.input_tensor_specs()?.values() {
            t.validate("full CNN input")?;
        }
        let (mut height, mut width) = (self.height, self.width);
        for level in 0..self.architecture.channels().len() {
            if height < 2 || width < 2 {
                return Err(invalid(
                    "a residual level cannot apply literal reflect padding to a singleton image",
                ));
            }
            let stride = self.architecture.spatial_downsample_factors()[level];
            if stride > 1 || self.architecture.temporal_downsample_factors()[level] > 1 {
                let pad = if stride == 2 { 1 } else { 0 };
                let extent = |n: u64| {
                    n.checked_add(pad)
                        .and_then(|n| n.checked_sub(3))
                        .map(|n| n / stride + 1)
                        .ok_or_else(|| {
                            invalid("single-frame downsample kernel exceeds padded spatial input")
                        })
                };
                height = extent(height)?;
                width = extent(width)?;
            }
        }
        if height < 2 || width < 2 {
            return Err(invalid(
                "final 3x3 reflect convolution requires at least two spatial values",
            ));
        }
        self.plan_nodes()?;
        Ok(())
    }
    pub(super) fn plan_nodes(&self) -> Result<usize, VNextError> {
        let a = &self.architecture;
        let blocks = usize::try_from(a.layers_per_block())
            .map_err(|_| invalid("block count exceeds address space"))?;
        let mut total = 4usize;
        for (level, &width) in a.channels().iter().enumerate() {
            total = total
                .checked_add(
                    blocks
                        .checked_mul(5)
                        .ok_or_else(|| invalid("node count overflows"))?,
                )
                .ok_or_else(|| invalid("node count overflows"))?;
            if level > 0 && a.channels()[level - 1] != width {
                total = total
                    .checked_add(1)
                    .ok_or_else(|| invalid("node count overflows"))?;
            }
            if a.spatial_downsample_factors()[level] > 1
                || a.temporal_downsample_factors()[level] > 1
            {
                total = total
                    .checked_add(1)
                    .ok_or_else(|| invalid("node count overflows"))?;
            }
        }
        Ok(total)
    }
}
pub(super) fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
