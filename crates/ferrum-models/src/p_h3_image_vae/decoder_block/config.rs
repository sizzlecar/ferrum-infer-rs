use super::*;
use serde::Deserializer;

/// Ferrum's independently declared numeric route. This is not the default
/// upstream FP16 route or its opportunistic INT8/BF16 attention implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageVaeDecoderBlockExecutionPreset {
    F32A8FullAttention,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageVaeDecoderBlockConfig {
    architecture: PH3ImageVaeDecoderConfig,
    block_index: u64,
    batch: u64,
    sequence_length: u64,
    execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    architecture: PH3ImageVaeDecoderConfig,
    block_index: u64,
    batch: u64,
    sequence_length: u64,
    execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
}
impl<'de> Deserialize<'de> for PH3ImageVaeDecoderBlockConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(
            w.architecture,
            w.block_index,
            w.batch,
            w.sequence_length,
            w.execution_preset,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl PH3ImageVaeDecoderBlockConfig {
    pub fn new(
        architecture: PH3ImageVaeDecoderConfig,
        block_index: u64,
        batch: u64,
        sequence_length: u64,
        execution_preset: PH3ImageVaeDecoderBlockExecutionPreset,
    ) -> Result<Self, VNextError> {
        let c = Self {
            architecture,
            block_index,
            batch,
            sequence_length,
            execution_preset,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn architecture(&self) -> &PH3ImageVaeDecoderConfig {
        &self.architecture
    }
    pub const fn block_index(&self) -> u64 {
        self.block_index
    }
    pub const fn batch(&self) -> u64 {
        self.batch
    }
    pub const fn sequence_length(&self) -> u64 {
        self.sequence_length
    }
    pub const fn execution_preset(&self) -> PH3ImageVaeDecoderBlockExecutionPreset {
        self.execution_preset
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.batch
            .checked_mul(self.sequence_length)
            .ok_or_else(|| invalid("layout", "row count overflows"))
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        self.validate()?;
        [
            (
                PH3ImageVaeDecoderBlockFamily::INPUT_ID,
                self.architecture.hidden_size()?,
            ),
            (
                PH3ImageVaeDecoderBlockFamily::COSINE_ID,
                self.architecture.rotary_dim(),
            ),
            (
                PH3ImageVaeDecoderBlockFamily::SINE_ID,
                self.architecture.rotary_dim(),
            ),
        ]
        .into_iter()
        .map(|(name, width)| {
            Ok((
                ProgramValueId::new(name)?,
                ProgramTensorSpec {
                    dimensions: vec![self.rows()?, width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            ))
        })
        .collect()
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let mut options = ProgramPlanCompileOptions::new(self.input_tensor_specs()?)?;
        for (value, tensor) in program::output_tensor_specs(self)? {
            options.insert_tensor_spec(value, tensor)?;
        }
        Ok(options)
    }
    pub(super) fn scale(&self) -> Result<CanonicalRational, VNextError> {
        let value = (1.0 / (self.architecture.head_dim() as f64).sqrt()) as f32;
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 255) as i32 - 150;
        CanonicalRational::new(
            i64::from((bits & 0x7fffff) | (1 << 23)),
            1u64.checked_shl(
                u32::try_from(-exponent)
                    .map_err(|_| invalid("attention", "scale exponent overflows"))?,
            )
            .ok_or_else(|| invalid("attention", "scale denominator overflows"))?,
        )
    }
    pub(super) fn attention_attributes(
        &self,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut a = BTreeMap::from([
            (
                AttributeId::new("sequence_length")?,
                SemanticValue::Unsigned(self.sequence_length),
            ),
            (
                AttributeId::new("head_dim")?,
                SemanticValue::Unsigned(self.architecture.head_dim()),
            ),
            (
                AttributeId::new("scale")?,
                SemanticValue::Rational(self.scale()?),
            ),
        ]);
        for (name, values) in [
            ("segment_lengths", vec![self.sequence_length as i64]),
            ("segment_bidirectional", vec![1]),
            ("key_run_counts", vec![1; self.batch as usize]),
            (
                "key_run_lengths",
                vec![self.sequence_length as i64; self.batch as usize],
            ),
            ("key_run_valid", vec![1; self.batch as usize]),
        ] {
            a.insert(AttributeId::new(name)?, SemanticValue::Integers(values));
        }
        Ok(a)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.architecture.validate()?;
        let a = &self.architecture;
        let h = a.hidden_size()?;
        let m = a.intermediate_size()?;
        if self.block_index >= a.layers()
            || self.batch == 0
            || self.batch > STRUCTURED_ATTENTION_BF16_MAX_BATCH as u64
            || self.sequence_length == 0
            || self.sequence_length > u64::from(u32::MAX)
            || !(2..=256).contains(&a.head_dim())
            || !a.head_dim().is_power_of_two()
            || h % 256 != 0
            || m % 256 != 0
        {
            return Err(invalid("config","requires an existing block, bounded positive batch/sequence, power-of-two head_dim in 2..256 and complete H256 projection inputs"));
        }
        let numerator = i128::from(a.norm_epsilon().numerator());
        let denominator = i128::from(a.norm_epsilon().denominator());
        if numerator * 1_000_000_000_000 < denominator {
            return Err(invalid(
                "epsilon",
                "executable RMS requires epsilon in [1e-12,1]",
            ));
        }
        let x = ResolvedTensorSpec::new(
            vec![self.rows()?, h],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        StructuredGqaAttentionF32Geometry::new(&x, &x, &x, &x, &self.attention_attributes()?)?;
        Ok(())
    }
}
