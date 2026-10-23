//! Separate wire names prevent reinterpreting the existing F16 evidence as BF16.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActualEncoding {
    F16,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NumericWeightQualityArtifactCase {
    pub case_id: String,
    pub actual_encoding: ActualEncoding,
    pub actual_le_sha256: String,
    pub actual_bits: Vec<u16>,
    pub reference_f32le_sha256: String,
    pub reference_f32_bits: Vec<u32>,
    pub relative_l2_upper_bound: CanonicalRational,
    pub nan_count: u64,
    pub inf_count: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct F16Case {
    case_id: String,
    actual_f16le_sha256: String,
    actual_f16_bits: Vec<u16>,
    reference_f32le_sha256: String,
    reference_f32_bits: Vec<u32>,
    relative_l2_upper_bound: CanonicalRational,
    nan_count: u64,
    inf_count: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bf16Case {
    case_id: String,
    actual_bf16le_sha256: String,
    actual_bf16_bits: Vec<u16>,
    reference_f32le_sha256: String,
    reference_f32_bits: Vec<u32>,
    relative_l2_upper_bound: CanonicalRational,
    nan_count: u64,
    inf_count: u64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Wire {
    F16(F16Case),
    Bf16(Bf16Case),
}

impl<'de> Deserialize<'de> for NumericWeightQualityArtifactCase {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Wire::deserialize(deserializer)? {
            Wire::F16(case) => Self {
                case_id: case.case_id,
                actual_encoding: ActualEncoding::F16,
                actual_le_sha256: case.actual_f16le_sha256,
                actual_bits: case.actual_f16_bits,
                reference_f32le_sha256: case.reference_f32le_sha256,
                reference_f32_bits: case.reference_f32_bits,
                relative_l2_upper_bound: case.relative_l2_upper_bound,
                nan_count: case.nan_count,
                inf_count: case.inf_count,
            },
            Wire::Bf16(case) => Self {
                case_id: case.case_id,
                actual_encoding: ActualEncoding::Bf16,
                actual_le_sha256: case.actual_bf16le_sha256,
                actual_bits: case.actual_bf16_bits,
                reference_f32le_sha256: case.reference_f32le_sha256,
                reference_f32_bits: case.reference_f32_bits,
                relative_l2_upper_bound: case.relative_l2_upper_bound,
                nan_count: case.nan_count,
                inf_count: case.inf_count,
            },
        })
    }
}

impl Serialize for NumericWeightQualityArtifactCase {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize borrowed fields directly: no duplicate vector backing.
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NumericWeightQualityArtifactCase", 8)?;
        state.serialize_field("case_id", &self.case_id)?;
        let (digest, bits) = match self.actual_encoding {
            ActualEncoding::F16 => ("actual_f16le_sha256", "actual_f16_bits"),
            ActualEncoding::Bf16 => ("actual_bf16le_sha256", "actual_bf16_bits"),
        };
        state.serialize_field(digest, &self.actual_le_sha256)?;
        state.serialize_field(bits, &self.actual_bits)?;
        state.serialize_field("reference_f32le_sha256", &self.reference_f32le_sha256)?;
        state.serialize_field("reference_f32_bits", &self.reference_f32_bits)?;
        state.serialize_field("relative_l2_upper_bound", &self.relative_l2_upper_bound)?;
        state.serialize_field("nan_count", &self.nan_count)?;
        state.serialize_field("inf_count", &self.inf_count)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        encoding: ActualEncoding,
        actual: u16,
        reference: f32,
    ) -> NumericWeightQualityArtifactCase {
        NumericWeightQualityArtifactCase {
            case_id: "encoding-boundary".to_owned(),
            actual_encoding: encoding,
            actual_le_sha256: digest_little_endian_u16(&[actual]),
            actual_bits: vec![actual],
            reference_f32le_sha256: digest_little_endian_u32(&[reference.to_bits()]),
            reference_f32_bits: vec![reference.to_bits()],
            relative_l2_upper_bound: CanonicalRational::new(2, 1).unwrap(),
            nan_count: 0,
            inf_count: 0,
        }
    }

    #[test]
    fn bf16_quality_extrema_subtract_in_f64_without_overflow() {
        let maximum = f32::from_bits(0x7f7f_0000);
        let opposite = case(ActualEncoding::Bf16, 0xff7f, maximum);
        let (relative_l2, nan, inf) = recompute_relative_l2(&opposite).unwrap();
        assert_eq!(relative_l2, 2.0);
        assert_eq!((nan, inf), (0, 0));
        let subnormal = case(ActualEncoding::Bf16, 1, f32::from_bits(0x0001_0000));
        assert_eq!(recompute_relative_l2(&subnormal).unwrap(), (0.0, 0, 0));
    }

    #[test]
    fn identical_u16_bits_keep_distinct_f16_and_bf16_meanings() {
        let f16 = case(ActualEncoding::F16, 0x3f80, 1.0);
        let bf16 = case(ActualEncoding::Bf16, 0x3f80, 1.0);
        assert_eq!(recompute_relative_l2(&f16).unwrap(), (0.875, 0, 0));
        assert_eq!(recompute_relative_l2(&bf16).unwrap(), (0.0, 0, 0));
        let f16_wire = serde_json::to_value(&f16).unwrap();
        let bf16_wire = serde_json::to_value(&bf16).unwrap();
        assert!(f16_wire.get("actual_f16_bits").is_some());
        assert!(f16_wire.get("actual_bf16_bits").is_none());
        assert!(bf16_wire.get("actual_bf16_bits").is_some());
        assert!(bf16_wire.get("actual_f16_bits").is_none());
    }
}
