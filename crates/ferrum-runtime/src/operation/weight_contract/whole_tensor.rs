use super::{
    ElementType, QuantizationFormatId, QuantizationGrouping, QuantizationPacking, QuantizationSpec,
    VNextError,
};

/// Versioned storage ABI: one U8 byte per E4M3FN value (sign, four exponent
/// bits with bias 7, three mantissa bits, maximum finite magnitude 448), and
/// one F32 dequantization multiplier for the entire logical tensor.
///
/// Logical values are `decode_e4m3fn(payload) * scale`; scale is not its
/// reciprocal. Signed zero and subnormals belong to the encoding, exponent
/// 15/mantissa 7 encodes NaN, and there is no infinity encoding. A materializer
/// or payload consumer must separately validate a finite, positive scale.
///
/// This identity specifies storage and decoding, not a quantization algorithm,
/// activation quantization, accumulation precision, or numerical qualification.
/// Declaring it does not advertise an executable backend implementation.
pub const FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID: &str =
    "quantization.fp8.e4m3fn.whole_tensor.f32.v1";

impl QuantizationSpec {
    pub fn fp8_e4m3fn_whole_tensor_f32() -> Self {
        Self {
            format_id: QuantizationFormatId::new(FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID)
                .expect("static quantization format identifier is valid"),
            bits_per_weight: 8,
            grouping: QuantizationGrouping::WholeTensor,
            packing: QuantizationPacking::Linear,
            scale_type: ElementType::F32,
            zero_point_type: None,
        }
    }

    pub(super) fn validate_reserved_whole_tensor_format(&self) -> Result<(), VNextError> {
        if self.format_id.as_str() == FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID
            && self != &Self::fp8_e4m3fn_whole_tensor_f32()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "quantization format `{}` requires eight-bit linear E4M3FN, whole-tensor grouping, one F32 scale, and no zero point",
                    self.format_id
                ),
            });
        }
        Ok(())
    }
}
