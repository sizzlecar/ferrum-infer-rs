//! Shared checked F32 scalar reads and arithmetic for the DiT consumers.
//! Unit-scale recipes are shared by Q/K/V and output projection consumers.
use super::FastH3TransformerPackage;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod decoder;
pub use decoder::FastH3DecoderAlphaDerivation;

pub(super) fn validate_original_scalar(c: &WeightComponentSpec) -> Result<(), VNextError> {
    if c.role != WeightComponentRole::Metadata
        || c.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::F32,
            })
        || !c.dimensions.is_empty()
        || c.external_names.len() != 1
        || !c.required
    {
        return Err(invalid("requires original scalar F32 metadata"));
    }
    Ok(())
}

pub(super) fn original_scalar(
    package: &FastH3TransformerPackage,
    c: &WeightComponentSpec,
) -> Result<(u32, String), VNextError> {
    original_scalar_from_source(package, c)
}

fn original_scalar_from_source(
    source: &dyn WeightComponentSource,
    c: &WeightComponentSpec,
) -> Result<(u32, String), VNextError> {
    validate_original_scalar(c)?;
    let payload = source.component(c)?;
    if payload.element_type() != ElementType::F32
        || payload.bytes().len() != 4
        || payload.source_files().len() != 1
    {
        return Err(invalid("requires original F32 with one source file"));
    }
    Ok((
        u32::from_le_bytes(
            payload
                .bytes()
                .try_into()
                .map_err(|_| invalid("scalar byte width"))?,
        ),
        payload.source_file().into(),
    ))
}

pub(super) fn quotient(alpha_bits: u32, sf_bits: u32) -> Result<u32, VNextError> {
    let alpha = f32::from_bits(alpha_bits);
    let sf = f32::from_bits(sf_bits);
    let result = alpha / sf;
    if ![alpha, sf, result]
        .into_iter()
        .all(|v| v.is_finite() && v > 0.)
    {
        return Err(invalid("alpha/SF must yield a positive finite RN32 scalar"));
    }
    Ok(result.to_bits())
}

pub(super) fn managed_scalar(
    host: &HostMemoryAccount,
    bits: u32,
) -> Result<ManagedHostBytes, HostMemoryError> {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::WeightBacking, 4)?;
    bytes.as_mut().copy_from_slice(&bits.to_le_bytes());
    Ok(bytes.freeze())
}

pub(super) fn valid_source_file(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('/')
        && !s.contains('\\')
        && !s.split('/').any(|p| matches!(p, "" | "." | ".."))
}

fn invalid(reason: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 NVFP4 scalar: {reason}"),
    }
}

const ALPHA_PROFILE: &str = "fast_h3.nvfp4.stored_alpha_div_unit.rn_f32.v1";

/// Original F32 alpha divided by the explicit unit activation scale. The
/// derived component ID covers the full recipe, including source identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3UnitAlphaDerivation {
    source_alpha: WeightComponentSpec,
    source_file: String,
    stored_bits: u32,
    arithmetic_profile: String,
}
impl FastH3UnitAlphaDerivation {
    pub(super) fn from_source(
        package: &FastH3TransformerPackage,
        source_alpha: WeightComponentSpec,
    ) -> Result<Self, VNextError> {
        let (stored_bits, source_file) = original_scalar(package, &source_alpha)?;
        let result = Self {
            source_alpha,
            source_file,
            stored_bits,
            arithmetic_profile: ALPHA_PROFILE.into(),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn source_alpha(&self) -> &WeightComponentSpec {
        &self.source_alpha
    }
    pub fn source_file(&self) -> &str {
        &self.source_file
    }
    pub fn stored_bits(&self) -> u32 {
        self.stored_bits
    }
    pub fn result_bits(&self) -> Result<u32, VNextError> {
        quotient(self.stored_bits, 1.0f32.to_bits())
    }
    pub fn arithmetic_profile(&self) -> &str {
        &self.arithmetic_profile
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        validate_original_scalar(&self.source_alpha)?;
        if self.arithmetic_profile != ALPHA_PROFILE
            || !valid_source_file(&self.source_file)
            || self.result_bits()? != self.stored_bits
        {
            return Err(invalid("invalid unit-alpha recipe"));
        }
        Ok(())
    }
    pub(super) fn output_component(&self) -> Result<WeightComponentSpec, VNextError> {
        self.validate()?;
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(self).map_err(invalid)?)
        );
        Ok(WeightComponentSpec {
            id: WeightId::new(format!("derived.fast_h3.unit_alpha.{digest}"))?,
            role: WeightComponentRole::Values,
            external_names: self.source_alpha.external_names.clone(),
            dimensions: vec![1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        })
    }
}
