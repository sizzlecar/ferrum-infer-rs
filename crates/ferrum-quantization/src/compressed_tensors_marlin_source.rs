//! Exact compressed-tensors W4 adapter for the Marlin physical ABI.
//!
//! The supported checkpoint subset is intentionally narrow: pack-quantized
//! INT4 weights, fixed group size, asymmetric zero points, and no activation
//! quantization. Repacking is cold-path CPU work performed while static plan
//! resources are initialized.

use std::borrow::Cow;
use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, QuantizationSpec, VNextError, WeightComponentPayload, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_kernels::marlin_repack::{
    repack_compressed_tensors_zero_points_to_marlin, repack_gptq_to_marlin_bytes_into,
    repack_scales_to_marlin,
};
use ferrum_types::Result;
use half::f16;
use safetensors::Dtype;

use crate::safetensors_archive::{transcode_dense_bytes, SafetensorsArchive, SafetensorsTensor};

pub const COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID: &str =
    "quantization.marlin.compressed-tensors-int4-asymmetric";

/// Mmap-backed safetensors archive with an exact compressed-tensors-to-Marlin
/// adapter. Dense components keep the archive's zero-copy/transcode behavior.
pub struct CompressedTensorsMarlinSafetensorsSource {
    archive: SafetensorsArchive,
}

impl CompressedTensorsMarlinSafetensorsSource {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        SafetensorsArchive::open(model_dir).map(Self::new)
    }

    pub const fn new(archive: SafetensorsArchive) -> Self {
        Self { archive }
    }

    pub const fn archive(&self) -> &SafetensorsArchive {
        &self.archive
    }

    fn packed_values<'source>(
        &'source self,
        component: &WeightComponentSpec,
        quantization: &QuantizationSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let group_size = validate_quantization(component, quantization)?;
        let [packed_name, shape_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "packed values require ordered weight_packed and weight_shape sources",
            ));
        };
        let stem = packed_name
            .strip_suffix(".weight_packed")
            .unwrap_or_default();
        if stem.is_empty() || shape_name != &format!("{stem}.weight_shape") {
            return Err(invalid_component(
                component,
                "packed values and shape metadata must share one compressed-tensors stem",
            ));
        }
        let packed = self.tensor(component, packed_name)?;
        let shape = self.tensor(component, shape_name)?;
        let (n, k) = validate_shape_metadata(component, &shape)?;
        if packed.dtype() != Dtype::I32 || packed.shape() != [n as u64, (k / 8) as u64] {
            return Err(invalid_component(
                component,
                format!(
                    "weight_packed must be I32[{n}, {}], got {:?} {:?}",
                    k / 8,
                    packed.dtype(),
                    packed.shape()
                ),
            ));
        }
        if k % group_size != 0 || k % 16 != 0 || n % 64 != 0 {
            return Err(invalid_component(
                component,
                format!("logical [N={n}, K={k}] is not group/Marlin aligned"),
            ));
        }
        let expected_dimensions = [n as u64, (k / 2) as u64];
        if !has_unit_prefix_and_tail(&component.dimensions, &expected_dimensions) {
            return Err(invalid_component(
                component,
                format!(
                    "packed component shape {:?} must be {expected_dimensions:?}",
                    component.dimensions
                ),
            ));
        }

        let source = decode_i32(packed.bytes(), component, "weight_packed")?;
        let mut gptq_words = vec![0_i32; source.len()];
        for output in 0..n {
            for packed_input in 0..k / 8 {
                gptq_words[packed_input * n + output] = source[output * (k / 8) + packed_input];
            }
        }
        let expected_bytes = usize::try_from(component.physical_bytes()?)
            .map_err(|_| invalid_component(component, "packed byte count exceeds address space"))?;
        let mut bytes = vec![0_u8; expected_bytes];
        repack_gptq_to_marlin_bytes_into(&gptq_words, k, n, &mut bytes);
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            vec![
                packed.source_file().to_owned(),
                shape.source_file().to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::U8,
            bytes,
        )
    }

    fn scales<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "scales require exactly one weight_scale source",
            ));
        };
        if !external_name.ends_with(".weight_scale") {
            return Err(invalid_component(
                component,
                "scale source must end with .weight_scale",
            ));
        }
        let tensor = self.tensor(component, external_name)?;
        let [n, groups] = tensor.shape() else {
            return Err(invalid_component(
                component,
                format!(
                    "weight_scale must have shape [N, K/G], got {:?}",
                    tensor.shape()
                ),
            ));
        };
        let n = usize::try_from(*n)
            .map_err(|_| invalid_component(component, "scale N exceeds address space"))?;
        let groups = usize::try_from(*groups)
            .map_err(|_| invalid_component(component, "scale group count exceeds address space"))?;
        if !has_unit_prefix_and_tail(&component.dimensions, &[n as u64, groups as u64]) {
            return Err(invalid_component(
                component,
                "typed scale dimensions differ from the checkpoint header",
            ));
        }
        let source_type = tensor.element_type().ok_or_else(|| {
            invalid_component(
                component,
                format!("weight_scale has unsupported dtype {:?}", tensor.dtype()),
            )
        })?;
        let f16_bytes = transcode_dense_bytes(
            tensor.bytes(),
            source_type,
            ElementType::F16,
            external_name,
            None,
        )?;
        let source = decode_f16(&f16_bytes, component)?;
        let mut group_major = vec![f16::ZERO; source.len()];
        for output in 0..n {
            for group in 0..groups {
                group_major[group * n + output] = source[output * groups + group];
            }
        }
        let repacked = repack_scales_to_marlin(&group_major, groups, n, 1);
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            vec![tensor.source_file().to_owned()],
            component.dimensions.clone(),
            ElementType::F16,
            encode_f16(repacked),
        )
    }

    fn zero_points<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "zero points require exactly one weight_zero_point source",
            ));
        };
        if !external_name.ends_with(".weight_zero_point") {
            return Err(invalid_component(
                component,
                "zero-point source must end with .weight_zero_point",
            ));
        }
        let tensor = self.tensor(component, external_name)?;
        let [packed_n, groups] = tensor.shape() else {
            return Err(invalid_component(
                component,
                format!(
                    "weight_zero_point must have shape [N/8, K/G], got {:?}",
                    tensor.shape()
                ),
            ));
        };
        if tensor.dtype() != Dtype::I32 {
            return Err(invalid_component(
                component,
                format!("weight_zero_point must be I32, got {:?}", tensor.dtype()),
            ));
        }
        let packed_n = usize::try_from(*packed_n)
            .map_err(|_| invalid_component(component, "zero-point N exceeds address space"))?;
        let groups = usize::try_from(*groups).map_err(|_| {
            invalid_component(component, "zero-point group count exceeds address space")
        })?;
        if !has_unit_prefix_and_tail(&component.dimensions, &[groups as u64, packed_n as u64]) {
            return Err(invalid_component(
                component,
                "typed zero-point dimensions must be Marlin [K/G, N/8]",
            ));
        }
        let source = decode_i32(tensor.bytes(), component, "weight_zero_point")?;
        let repacked =
            repack_compressed_tensors_zero_points_to_marlin(&source, groups, packed_n * 8);
        let bytes = repacked
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            vec![tensor.source_file().to_owned()],
            component.dimensions.clone(),
            ElementType::I32,
            bytes,
        )
    }

    fn tensor<'source>(
        &'source self,
        component: &WeightComponentSpec,
        external_name: &str,
    ) -> std::result::Result<SafetensorsTensor<'source>, VNextError> {
        self.archive
            .tensor(external_name)
            .map_err(|error| invalid_component(component, error.to_string()))
    }
}

impl WeightComponentSource for CompressedTensorsMarlinSafetensorsSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        match (&component.role, &component.encoding) {
            (WeightComponentRole::PackedValues, WeightEncoding::Quantized(quantization)) => {
                self.packed_values(component, quantization)
            }
            (
                WeightComponentRole::Scales,
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
            ) => self.scales(component),
            (
                WeightComponentRole::ZeroPoints,
                WeightEncoding::Dense {
                    element_type: ElementType::I32,
                },
            ) => self.zero_points(component),
            (_, WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. }) => {
                self.archive.component(component)
            }
            _ => Err(invalid_component(
                component,
                "compressed-tensors Marlin adapter received an unsupported component encoding",
            )),
        }
    }
}

fn validate_quantization(
    component: &WeightComponentSpec,
    quantization: &QuantizationSpec,
) -> std::result::Result<usize, VNextError> {
    quantization.validate()?;
    let Some(group_size) = quantization.grouping.fixed_size() else {
        return Err(invalid_component(
            component,
            "compressed-tensors requires fixed-size groups",
        ));
    };
    if quantization.format_id.as_str() != COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type != Some(ElementType::I32)
    {
        return Err(invalid_component(
            component,
            "compressed-tensors requires asymmetric INT4 Marlin packing with F16 scales and packed I32 zero points",
        ));
    }
    usize::try_from(group_size)
        .map_err(|_| invalid_component(component, "group size exceeds address space"))
}

fn has_unit_prefix_and_tail(dimensions: &[u64], tail: &[u64; 2]) -> bool {
    dimensions.len() >= 2
        && dimensions[dimensions.len() - 2..] == *tail
        && dimensions[..dimensions.len() - 2]
            .iter()
            .all(|extent| *extent == 1)
}

fn validate_shape_metadata(
    component: &WeightComponentSpec,
    tensor: &SafetensorsTensor<'_>,
) -> std::result::Result<(usize, usize), VNextError> {
    if tensor.dtype() != Dtype::I64 || tensor.shape() != [2] || tensor.bytes().len() != 16 {
        return Err(invalid_component(component, "weight_shape must be I64[2]"));
    }
    let values = tensor
        .bytes()
        .chunks_exact(8)
        .map(|bytes| {
            i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect::<Vec<_>>();
    let n = usize::try_from(values[0])
        .map_err(|_| invalid_component(component, "weight_shape N must be positive"))?;
    let k = usize::try_from(values[1])
        .map_err(|_| invalid_component(component, "weight_shape K must be positive"))?;
    if n == 0 || k == 0 || !k.is_multiple_of(8) {
        return Err(invalid_component(
            component,
            "weight_shape must contain positive [N, K] with K divisible by 8",
        ));
    }
    Ok((n, k))
}

fn decode_i32(
    bytes: &[u8],
    component: &WeightComponentSpec,
    label: &str,
) -> std::result::Result<Vec<i32>, VNextError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid_component(
            component,
            format!("{label} byte length is not I32 aligned"),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn decode_f16(
    bytes: &[u8],
    component: &WeightComponentSpec,
) -> std::result::Result<Vec<f16>, VNextError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(invalid_component(
            component,
            "scale byte length is not F16 aligned",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|value| f16::from_le_bytes([value[0], value[1]]))
        .collect())
}

fn encode_f16(values: Vec<f16>) -> Cow<'static, [u8]> {
    Cow::Owned(
        values
            .into_iter()
            .flat_map(f16::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

fn invalid_component(component: &WeightComponentSpec, reason: impl AsRef<str>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!(
            "compressed-tensors component `{}` is invalid: {}",
            component.id,
            reason.as_ref()
        ),
    }
}
