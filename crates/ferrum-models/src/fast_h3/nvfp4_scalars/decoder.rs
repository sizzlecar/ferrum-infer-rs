use super::*;

const PROFILE: &str = "fast_h3.video_decoder.reciprocal_unit_times_weight_sf.rn_f32.v1";

/// Original packed decoder recipe, distinct from the DiT's stored alpha/unit.
/// The checkpoint's F32 global scale remains authoritative. This metadata is
/// not execution authority; a consumer must bind it to its selected package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3DecoderAlphaDerivation {
    source_global_scale: WeightComponentSpec,
    source_file: String,
    stored_bits: u32,
    arithmetic_profile: String,
}

impl FastH3DecoderAlphaDerivation {
    pub(crate) fn from_source(
        source: &dyn WeightComponentSource,
        source_global_scale: WeightComponentSpec,
    ) -> Result<Self, VNextError> {
        let (stored_bits, source_file) = original_scalar_from_source(source, &source_global_scale)?;
        let result = Self {
            source_global_scale,
            source_file,
            stored_bits,
            arithmetic_profile: PROFILE.into(),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn source_global_scale(&self) -> &WeightComponentSpec {
        &self.source_global_scale
    }
    pub fn source_file(&self) -> &str {
        &self.source_file
    }
    pub fn stored_bits(&self) -> u32 {
        self.stored_bits
    }
    pub fn arithmetic_profile(&self) -> &str {
        &self.arithmetic_profile
    }
    pub fn result_bits(&self) -> Result<u32, VNextError> {
        // Upstream unit activation scale is multiplied first in F32; the
        // reciprocal is rounded in F32, without consulting BF16 master weights.
        let denominator = 1.0f32 * f32::from_bits(self.stored_bits);
        quotient(1.0f32.to_bits(), denominator.to_bits())
    }
    fn validate(&self) -> Result<(), VNextError> {
        validate_original_scalar(&self.source_global_scale)?;
        if self.arithmetic_profile != PROFILE || !valid_source_file(&self.source_file) {
            return Err(invalid("invalid decoder unit-global reciprocal recipe"));
        }
        self.result_bits()?;
        Ok(())
    }
    pub fn output_component(&self) -> Result<WeightComponentSpec, VNextError> {
        self.validate()?;
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(self).map_err(invalid)?)
        );
        Ok(WeightComponentSpec {
            id: WeightId::new(format!("derived.fast_h3.decoder_alpha.{digest}"))?,
            role: WeightComponentRole::Values,
            external_names: self.source_global_scale.external_names.clone(),
            dimensions: vec![1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        })
    }
    /// Re-read and compare the full original recipe before charging four bytes
    /// to the caller's existing Host owner. A deserialized recipe cannot supply
    /// its own authority or substitute a same-shaped scalar from another file.
    pub fn managed_backing(
        &self,
        source: &dyn WeightComponentSource,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostBytes, VNextError> {
        self.validate()?;
        if *self != Self::from_source(source, self.source_global_scale.clone())? {
            return Err(invalid(
                "decoder alpha recipe differs from selected original source",
            ));
        }
        managed_scalar(host, self.result_bits()?).map_err(invalid)
    }
}
