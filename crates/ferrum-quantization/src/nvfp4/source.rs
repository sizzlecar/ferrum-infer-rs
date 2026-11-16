use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    ElementType, QuantizationPacking, VNextError, WeightComponentPayload, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_types::Result;
use safetensors::Dtype;

use super::{
    decode_e2m1, decode_e4m3_scale, invalid, Nvfp4Geometry, NVFP4_E2M1_E4M3_128X4_FORMAT_ID,
};
use crate::{SafetensorsArchive, SafetensorsTensor};

/// Explicit physical names: model packages select projections and source
/// conventions. No layer-number or model-name rule selects quantization.
#[derive(Clone, Debug)]
pub struct Nvfp4MatrixNames {
    pub packed: String,
    pub scales: String,
    pub global_scale: String,
    /// The serialized FP32 output multiplier is authoritative when present.
    /// FastVideo DiT stores its global scale in BF16 but preserves this FP32
    /// reciprocal separately; recomputing it changes arithmetic.
    pub output_multiplier: Option<String>,
    pub input_global_scale: Option<String>,
}

impl Nvfp4MatrixNames {
    pub fn fastvideo_dit(prefix: &str, static_input_scale: bool) -> Self {
        Self {
            packed: format!("{prefix}::_nvfp4_weight"),
            scales: format!("{prefix}::_nvfp4_weight_scale"),
            global_scale: format!("{prefix}::_weight_global_sf"),
            output_multiplier: Some(format!("{prefix}::_nvfp4_alpha")),
            input_global_scale: static_input_scale
                .then(|| format!("{prefix}::_nvfp4_input_global_sf")),
        }
    }
    pub fn serialized_encoder(prefix: &str) -> Self {
        Self {
            packed: format!("{prefix}.weight_packed"),
            scales: format!("{prefix}.weight_scale"),
            global_scale: format!("{prefix}.weight_global_scale"),
            output_multiplier: None,
            input_global_scale: None,
        }
    }
    pub fn packed_decoder(prefix: &str) -> Self {
        Self {
            packed: format!("{prefix}.packed_weight"),
            scales: format!("{prefix}.weight_inv_scale"),
            global_scale: format!("{prefix}.weight_global_sf"),
            output_multiplier: None,
            input_global_scale: None,
        }
    }
    fn all(&self) -> impl Iterator<Item = &str> {
        [&self.packed, &self.scales, &self.global_scale]
            .into_iter()
            .map(String::as_str)
            .chain(self.output_multiplier.as_deref())
            .chain(self.input_global_scale.as_deref())
    }
}

#[derive(Clone, Debug)]
pub struct Nvfp4MatrixSpec {
    pub names: Nvfp4MatrixNames,
    pub geometry: Nvfp4Geometry,
}

/// Bounds the declared inventory and mapped tensor payload being validated.
/// This is not a GPU/Host allocation grant or a bound on archive metadata.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4SourceLimits {
    pub maximum_matrices: usize,
    pub maximum_source_bytes: usize,
}

#[derive(Clone, Copy)]
enum Part {
    Packed,
    Scales,
    Scalar,
}
struct MatrixRecord {
    spec: Nvfp4MatrixSpec,
    output_multiplier: f32,
    input_global_scale: Option<f32>,
}

/// Validates explicitly declared matrices without allocating dense weights.
/// The archive's immutable mmap ownership is retained by every payload.
pub struct Nvfp4SafetensorsSource {
    archive: std::sync::Arc<SafetensorsArchive>,
    matrices: BTreeMap<String, MatrixRecord>,
    parts: BTreeMap<String, Part>,
}

impl Nvfp4SafetensorsSource {
    pub fn new(
        archive: SafetensorsArchive,
        specs: Vec<Nvfp4MatrixSpec>,
        limits: Nvfp4SourceLimits,
    ) -> Result<Self> {
        Self::from_shared_archive(std::sync::Arc::new(archive), specs, limits)
    }

    /// Reuse the original immutable mapping across shape-specific model plans.
    /// Payload admission and retained physical source identity are unchanged.
    pub fn from_shared_archive(
        archive: std::sync::Arc<SafetensorsArchive>,
        specs: Vec<Nvfp4MatrixSpec>,
        limits: Nvfp4SourceLimits,
    ) -> Result<Self> {
        if specs.is_empty()
            || specs.len() > limits.maximum_matrices
            || limits.maximum_source_bytes == 0
        {
            return Err(invalid(
                "matrix inventory exceeds declared limits or is empty",
            ));
        }
        let mut source_bytes = 0usize;
        let mut names = BTreeSet::new();
        // Check all byte limits before walking any scale payload.
        for spec in &specs {
            for name in spec.names.all() {
                if name.is_empty() || !names.insert(name) {
                    return Err(invalid(
                        "component names must be nonempty and unique across matrices",
                    ));
                }
                let tensor = archive.tensor(name)?;
                source_bytes = source_bytes
                    .checked_add(tensor.bytes().len())
                    .filter(|n| *n <= limits.maximum_source_bytes)
                    .ok_or_else(|| invalid("declared tensor bytes exceed source limit"))?;
            }
        }
        let mut matrices = BTreeMap::new();
        let mut parts = BTreeMap::new();
        for spec in specs {
            let packed = archive.tensor(&spec.names.packed)?;
            let scales = archive.tensor(&spec.names.scales)?;
            let geometry = spec.geometry;
            let packed_shape = geometry.packed_shape().map(|n| n as u64);
            if packed.dtype() != Dtype::U8
                || packed.shape() != packed_shape
                || packed.bytes().len() != geometry.packed_bytes()
            {
                return Err(invalid(
                    "packed tensor must be U8 [out, in/2] matching declared geometry",
                ));
            }
            // FlashInfer can label identical scale bytes U8 or F8_E4M3 and
            // stores the swizzled physical buffer flattened or as a matrix.
            let scale_shape = geometry.scale_shape().map(|n| n as u64);
            if !matches!(scales.dtype(), Dtype::U8 | Dtype::F8_E4M3)
                || !(scales.shape() == scale_shape
                    || scales.shape() == [geometry.scale_bytes() as u64])
                || scales.bytes().len() != geometry.scale_bytes()
            {
                return Err(invalid(
                    "scale tensor must be exact U8/E4M3 128x4 padded storage",
                ));
            }
            geometry.validate_scales(scales.bytes())?;
            let global = scalar(&archive.tensor(&spec.names.global_scale)?, true)?;
            let output_multiplier = match &spec.names.output_multiplier {
                Some(name) => scalar(&archive.tensor(name)?, false)?,
                None => {
                    let reciprocal = (1.0f64 / f64::from(global)) as f32;
                    positive(reciprocal)?
                }
            };
            let input_global_scale = spec
                .names
                .input_global_scale
                .as_ref()
                .map(|name| scalar(&archive.tensor(name)?, false))
                .transpose()?;
            parts.insert(spec.names.packed.clone(), Part::Packed);
            parts.insert(spec.names.scales.clone(), Part::Scales);
            parts.insert(spec.names.global_scale.clone(), Part::Scalar);
            for name in [
                &spec.names.output_multiplier,
                &spec.names.input_global_scale,
            ]
            .into_iter()
            .flatten()
            {
                parts.insert(name.clone(), Part::Scalar);
            }
            matrices.insert(
                spec.names.packed.clone(),
                MatrixRecord {
                    spec,
                    output_multiplier,
                    input_global_scale,
                },
            );
        }
        Ok(Self {
            archive,
            matrices,
            parts,
        })
    }

    pub fn archive(&self) -> &SafetensorsArchive {
        &self.archive
    }
    pub fn matrix(&self, packed_name: &str) -> Result<Nvfp4Matrix<'_>> {
        let record = self
            .matrices
            .get(packed_name)
            .ok_or_else(|| invalid("matrix was not declared"))?;
        Ok(Nvfp4Matrix {
            packed: self.archive.tensor(&record.spec.names.packed)?,
            scales: self.archive.tensor(&record.spec.names.scales)?,
            geometry: record.spec.geometry,
            output_multiplier: record.output_multiplier,
            input_global_scale: record.input_global_scale,
        })
    }
}

fn positive(value: f32) -> Result<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(
            "global scale or multiplier must be finite and positive",
        ));
    }
    Ok(value)
}

fn scalar(tensor: &SafetensorsTensor<'_>, allow_bf16: bool) -> Result<f32> {
    if !(tensor.shape().is_empty() || tensor.shape() == [1]) {
        return Err(invalid(
            "global scale or multiplier requires scalar or [1] storage",
        ));
    }
    let value = match (tensor.dtype(), tensor.bytes()) {
        (Dtype::F32, [a, b, c, d]) => f32::from_le_bytes([*a, *b, *c, *d]),
        (Dtype::BF16, [a, b]) if allow_bf16 => {
            half::bf16::from_bits(u16::from_le_bytes([*a, *b])).to_f32()
        }
        _ => {
            return Err(invalid(
                "global scale requires F32/BF16; explicit output/input multiplier requires F32",
            ))
        }
    };
    positive(value)
}

pub struct Nvfp4Matrix<'a> {
    packed: SafetensorsTensor<'a>,
    scales: SafetensorsTensor<'a>,
    geometry: Nvfp4Geometry,
    output_multiplier: f32,
    input_global_scale: Option<f32>,
}
impl Nvfp4Matrix<'_> {
    pub const fn geometry(&self) -> Nvfp4Geometry {
        self.geometry
    }
    pub fn packed_bytes(&self) -> &[u8] {
        self.packed.bytes()
    }
    pub fn scale_bytes(&self) -> &[u8] {
        self.scales.bytes()
    }
    pub const fn output_multiplier(&self) -> f32 {
        self.output_multiplier
    }
    /// None means the source did not declare an activation scale. Runtime
    /// policy must explicitly choose unit/dynamic/calibrated behavior.
    pub const fn input_global_scale(&self) -> Option<f32> {
        self.input_global_scale
    }
    pub fn value(&self, row: usize, column: usize) -> Result<f32> {
        if row >= self.geometry.rows() || column >= self.geometry.columns() {
            return Err(invalid("matrix coordinate out of bounds"));
        }
        let byte = self.packed.bytes()[row * (self.geometry.columns() / 2) + column / 2];
        let code = (byte >> ((column % 2) * 4)) & 15;
        let scale =
            decode_e4m3_scale(self.scales.bytes()[self.geometry.scale_index(row, column / 16)?])?;
        let value = (decode_e2m1(code)? * scale) * self.output_multiplier;
        if !value.is_finite() {
            return Err(invalid("dequantized value overflows FP32"));
        }
        Ok(value)
    }
    /// Mathematical FP32 weight reference, not the deployed quantized GEMM.
    /// Caller owns storage; errors are checked before the first output write.
    pub fn dequantize_f32_into(&self, output: &mut [f32]) -> Result<()> {
        if output.len() != self.geometry.element_count() {
            return Err(invalid("dense output length differs from logical matrix"));
        }
        for row in 0..self.geometry.rows() {
            for column in 0..self.geometry.columns() {
                self.value(row, column)?;
            }
        }
        for (index, value) in output.iter_mut().enumerate() {
            *value = self.value(
                index / self.geometry.columns(),
                index % self.geometry.columns(),
            )?;
        }
        Ok(())
    }
}

impl WeightComponentSource for Nvfp4SafetensorsSource {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        let fail = |reason: &str| VNextError::InvalidExecutionPlan {
            reason: format!("NVFP4 component {}: {reason}", component.id),
        };
        let [name] = component.external_names.as_slice() else {
            return Err(fail("one exact external tensor is required"));
        };
        let Some(part) = self.parts.get(name) else {
            return match (&component.role, &component.encoding) {
                (WeightComponentRole::PackedValues, _) | (_, WeightEncoding::Quantized(_)) => {
                    Err(fail("quantized tensor was not validated in this source"))
                }
                _ => self.archive.component(component),
            };
        };
        let tensor = self
            .archive
            .tensor(name)
            .map_err(|e| fail(&e.to_string()))?;
        let element_type = match part {
            Part::Packed => {
                let WeightEncoding::Quantized(q) = &component.encoding else {
                    return Err(fail("packed values require NVFP4 encoding"));
                };
                q.validate()?;
                if component.role != WeightComponentRole::PackedValues
                    || q.format_id.as_str() != NVFP4_E2M1_E4M3_128X4_FORMAT_ID
                    || q.bits_per_weight != 4
                    || q.grouping.fixed_size() != Some(16)
                    || q.packing != QuantizationPacking::Interleaved
                    || q.scale_type != ElementType::U8
                    || q.zero_point_type.is_some()
                {
                    return Err(fail(
                        "packed quantization differs from E2M1/group16/E4M3/128x4 contract",
                    ));
                }
                ElementType::U8
            }
            Part::Scales => {
                if component.role != WeightComponentRole::Scales
                    || component.encoding
                        != (WeightEncoding::Dense {
                            element_type: ElementType::U8,
                        })
                {
                    return Err(fail("scales require dense U8 physical bytes"));
                }
                ElementType::U8
            }
            Part::Scalar => {
                let expected = tensor
                    .element_type()
                    .ok_or_else(|| fail("unsupported scalar dtype"))?;
                if component.encoding
                    != (WeightEncoding::Dense {
                        element_type: expected,
                    })
                {
                    return Err(fail("scalar encoding must preserve source dtype"));
                }
                expected
            }
        };
        if component.dimensions != tensor.shape() {
            return Err(fail("physical shape differs from source"));
        }
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            element_type,
            tensor.bytes(),
        )?
        .with_retained_host_memory(tensor.retained_host_memory().clone())
    }
}
