//! Exact compressed-tensors W4 adapter for the Marlin physical ABI.
//!
//! The supported checkpoint subset is intentionally narrow: pack-quantized
//! INT4 weights, fixed group size, optional format-typed asymmetric zero
//! points, and no activation quantization. Repacking is cold-path CPU work
//! performed while static plan resources are initialized.

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
pub const COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID: &str =
    "quantization.marlin.compressed-tensors-int4-symmetric";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompressedTensorsInt4Mode {
    Asymmetric,
    Symmetric,
}

struct ValidatedQuantization {
    group_size: usize,
    mode: CompressedTensorsInt4Mode,
}

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
        let validated = validate_quantization(component, quantization)?;
        let group_size = validated.group_size;
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
        if validated.mode == CompressedTensorsInt4Mode::Symmetric
            && self.archive.contains(&format!("{stem}.weight_zero_point"))
        {
            return Err(invalid_component(
                component,
                "symmetric compressed-tensors must not provide weight_zero_point",
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
        // compressed-tensors pack-quantized INT4 adds the signed-domain bias
        // of eight before packing and stores the first input in the low
        // nibble. Those codes are already Marlin U4B8 codes, so symmetric
        // weights need only the [N, K/8] -> [K/8, N] transpose above and the
        // normal Marlin tile permutation. Applying another bias/XOR here
        // would corrupt the signed-domain meaning of every symmetric code.
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
        let stem = external_name
            .strip_suffix(".weight_scale")
            .unwrap_or_default();
        if stem.is_empty() {
            return Err(invalid_component(
                component,
                "scale source must have a non-empty compressed-tensors stem",
            ));
        }
        let tensor = self.tensor(component, external_name)?;
        // Scale components are dense and therefore do not carry the packed
        // component's QuantizationSpec. Absence of the same-stem zero-point
        // sidecar is the physical symmetric-bundle discriminator. Keep the
        // established asymmetric transcode path unchanged, but require the
        // exact BF16[N, K/32] header for a zero-point-free bundle.
        if !self.archive.contains(&format!("{stem}.weight_zero_point")) {
            let shape = self.tensor(component, &format!("{stem}.weight_shape"))?;
            let (shape_n, shape_k) = validate_shape_metadata(component, &shape)?;
            if tensor.dtype() != Dtype::BF16
                || !shape_k.is_multiple_of(32)
                || tensor.shape() != [shape_n as u64, (shape_k / 32) as u64]
            {
                return Err(invalid_component(
                    component,
                    format!(
                        "symmetric weight_scale must be BF16[{shape_n}, {}], got {:?} {:?}",
                        shape_k / 32,
                        tensor.dtype(),
                        tensor.shape()
                    ),
                ));
            }
        }
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
) -> std::result::Result<ValidatedQuantization, VNextError> {
    quantization.validate()?;
    let Some(group_size) = quantization.grouping.fixed_size() else {
        return Err(invalid_component(
            component,
            "compressed-tensors requires fixed-size groups",
        ));
    };
    let mode = match quantization.format_id.as_str() {
        COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID
            if quantization.bits_per_weight == 4
                && quantization.scale_type == ElementType::F16
                && quantization.zero_point_type == Some(ElementType::I32) =>
        {
            CompressedTensorsInt4Mode::Asymmetric
        }
        COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID
            if quantization.bits_per_weight == 4
                && group_size == 32
                && quantization.packing == ferrum_interfaces::vnext::QuantizationPacking::Tiled
                && quantization.scale_type == ElementType::F16
                && quantization.zero_point_type.is_none() =>
        {
            CompressedTensorsInt4Mode::Symmetric
        }
        COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID => {
            return Err(invalid_component(
                component,
                "asymmetric compressed-tensors requires INT4 Marlin packing with F16 scales and packed I32 zero points",
            ));
        }
        COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID => {
            return Err(invalid_component(
                component,
                "symmetric compressed-tensors requires group32 INT4 tiled Marlin packing with F16 scales and no zero points",
            ));
        }
        _ => {
            return Err(invalid_component(
                component,
                "compressed-tensors quantization format id is unsupported",
            ));
        }
    };
    Ok(ValidatedQuantization {
        group_size: usize::try_from(group_size)
            .map_err(|_| invalid_component(component, "group size exceeds address space"))?,
        mode,
    })
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ferrum_interfaces::vnext::{
        QuantizationFormatId, QuantizationGrouping, QuantizationPacking, WeightId,
    };
    use half::bf16;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    const STEM: &str = "model.layers.0.self_attn.q_proj";

    struct Fixture {
        directory: tempfile::TempDir,
        packed: Vec<i32>,
        scales: Vec<bf16>,
        zero_points: Vec<i32>,
        n: usize,
        k: usize,
        groups: usize,
    }

    fn write_fixture() -> Fixture {
        write_fixture_with_options(true, Dtype::BF16)
    }

    fn write_symmetric_fixture() -> Fixture {
        write_fixture_with_options(false, Dtype::BF16)
    }

    fn write_fixture_with_options(include_zero_points: bool, scale_dtype: Dtype) -> Fixture {
        assert!(matches!(scale_dtype, Dtype::BF16 | Dtype::F16));
        let directory = tempdir().unwrap();
        let n = 64_usize;
        let k = 128_usize;
        let groups = k / 32;
        let packed = (0..n)
            .flat_map(|output| {
                (0..k / 8).map(move |packed_input| {
                    (0..8).fold(0_u32, |word, lane| {
                        let input = packed_input * 8 + lane;
                        let value = ((output * 3 + input * 5 + 1) % 16) as u32;
                        word | (value << (lane * 4))
                    }) as i32
                })
            })
            .collect::<Vec<_>>();
        let scales = (0..n)
            .flat_map(|output| {
                (0..groups).map(move |group| {
                    bf16::from_f32(0.015625 * (1 + (output + group * 7) % 11) as f32)
                })
            })
            .collect::<Vec<_>>();
        let zero_points = (0..n / 8)
            .flat_map(|packed_output| {
                (0..groups).map(move |group| {
                    (0..8).fold(0_u32, |word, lane| {
                        let output = packed_output * 8 + lane;
                        let value = ((output + group * 3 + 2) % 15) as u32;
                        word | (value << (lane * 4))
                    }) as i32
                })
            })
            .collect::<Vec<_>>();
        let packed_bytes = packed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let scale_bytes = scales
            .iter()
            .flat_map(|value| {
                if scale_dtype == Dtype::BF16 {
                    value.to_bits().to_le_bytes()
                } else {
                    f16::from_f32(value.to_f32()).to_bits().to_le_bytes()
                }
            })
            .collect::<Vec<_>>();
        let zero_point_bytes = zero_points
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let shape_bytes = [n as i64, k as i64]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>();
        let mut views = BTreeMap::from([
            (
                format!("{STEM}.weight_packed"),
                TensorView::new(Dtype::I32, vec![n, k / 8], &packed_bytes).unwrap(),
            ),
            (
                format!("{STEM}.weight_scale"),
                TensorView::new(scale_dtype, vec![n, groups], &scale_bytes).unwrap(),
            ),
            (
                format!("{STEM}.weight_shape"),
                TensorView::new(Dtype::I64, vec![2], &shape_bytes).unwrap(),
            ),
        ]);
        if include_zero_points {
            views.insert(
                format!("{STEM}.weight_zero_point"),
                TensorView::new(Dtype::I32, vec![n / 8, groups], &zero_point_bytes).unwrap(),
            );
        }
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        Fixture {
            directory,
            packed,
            scales,
            zero_points,
            n,
            k,
            groups,
        }
    }

    fn quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID).unwrap(),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(32),
            packing: QuantizationPacking::Tiled,
            scale_type: ElementType::F16,
            zero_point_type: Some(ElementType::I32),
        }
    }

    fn symmetric_quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(
                COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID,
            )
            .unwrap(),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(32),
            packing: QuantizationPacking::Tiled,
            scale_type: ElementType::F16,
            zero_point_type: None,
        }
    }

    #[test]
    fn repacks_all_compressed_tensors_sidecars_at_the_source_boundary() {
        let fixture = write_fixture();
        let source =
            CompressedTensorsMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        let packed_component = WeightComponentSpec {
            id: WeightId::new("component.q.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                format!("{STEM}.weight_packed"),
                format!("{STEM}.weight_shape"),
            ],
            dimensions: vec![1, fixture.n as u64, (fixture.k / 2) as u64],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        };
        let packed_payload = source.component(&packed_component).unwrap();
        let mut gptq_words = vec![0_i32; fixture.packed.len()];
        for output in 0..fixture.n {
            for packed_input in 0..fixture.k / 8 {
                gptq_words[packed_input * fixture.n + output] =
                    fixture.packed[output * (fixture.k / 8) + packed_input];
            }
        }
        let mut expected_packed = vec![0_u8; fixture.n * fixture.k / 2];
        repack_gptq_to_marlin_bytes_into(&gptq_words, fixture.k, fixture.n, &mut expected_packed);
        assert_eq!(packed_payload.bytes(), expected_packed);
        assert_eq!(packed_payload.dimensions(), packed_component.dimensions);

        let scales_component = WeightComponentSpec {
            id: WeightId::new("component.q.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec![format!("{STEM}.weight_scale")],
            dimensions: vec![1, fixture.n as u64, fixture.groups as u64],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let scales_payload = source.component(&scales_component).unwrap();
        let mut group_major = vec![f16::ZERO; fixture.scales.len()];
        for output in 0..fixture.n {
            for group in 0..fixture.groups {
                group_major[group * fixture.n + output] =
                    f16::from_f32(fixture.scales[output * fixture.groups + group].to_f32());
            }
        }
        let expected_scales = encode_f16(repack_scales_to_marlin(
            &group_major,
            fixture.groups,
            fixture.n,
            1,
        ));
        assert_eq!(scales_payload.bytes(), expected_scales.as_ref());

        let zero_points_component = WeightComponentSpec {
            id: WeightId::new("component.q.zero_points").unwrap(),
            role: WeightComponentRole::ZeroPoints,
            external_names: vec![format!("{STEM}.weight_zero_point")],
            dimensions: vec![1, fixture.groups as u64, (fixture.n / 8) as u64],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::I32,
            },
            required: true,
        };
        let zero_points_payload = source.component(&zero_points_component).unwrap();
        let expected_zero_points = repack_compressed_tensors_zero_points_to_marlin(
            &fixture.zero_points,
            fixture.groups,
            fixture.n,
        )
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
        assert_eq!(zero_points_payload.bytes(), expected_zero_points);
    }

    #[test]
    fn repacks_symmetric_group32_codes_as_marlin_u4b8_without_a_zero_point() {
        let fixture = write_symmetric_fixture();
        let source =
            CompressedTensorsMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        assert!(!source
            .archive()
            .contains(&format!("{STEM}.weight_zero_point")));

        // pack_to_int32 stores signed q in the biased code domain q + 8 and
        // puts input lane zero in the low nibble. Check the fixture itself so
        // the expected Marlin bytes below catch any accidental second bias.
        let expected_first_word = (0..8).fold(0_u32, |word, lane| {
            let signed = ((lane * 5 + 1) % 16) as i32 - 8;
            word | ((signed + 8) as u32) << (lane * 4)
        });
        assert_eq!(fixture.packed[0] as u32, expected_first_word);

        let packed_component = WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                format!("{STEM}.weight_packed"),
                format!("{STEM}.weight_shape"),
            ],
            dimensions: vec![fixture.n as u64, (fixture.k / 2) as u64],
            encoding: WeightEncoding::Quantized(symmetric_quantization()),
            required: true,
        };
        let packed_payload = source.component(&packed_component).unwrap();
        let mut direct_u4b8_words = vec![0_i32; fixture.packed.len()];
        for output in 0..fixture.n {
            for packed_input in 0..fixture.k / 8 {
                direct_u4b8_words[packed_input * fixture.n + output] =
                    fixture.packed[output * (fixture.k / 8) + packed_input];
            }
        }
        let mut expected_packed = vec![0_u8; fixture.n * fixture.k / 2];
        repack_gptq_to_marlin_bytes_into(
            &direct_u4b8_words,
            fixture.k,
            fixture.n,
            &mut expected_packed,
        );
        assert_eq!(packed_payload.bytes(), expected_packed);
        assert_eq!(packed_payload.element_type(), ElementType::U8);

        let scales_component = WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec![format!("{STEM}.weight_scale")],
            dimensions: vec![fixture.n as u64, fixture.groups as u64],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let scales_payload = source.component(&scales_component).unwrap();
        let mut group_major = vec![f16::ZERO; fixture.scales.len()];
        for output in 0..fixture.n {
            for group in 0..fixture.groups {
                group_major[group * fixture.n + output] =
                    f16::from_f32(fixture.scales[output * fixture.groups + group].to_f32());
            }
        }
        let expected_scales = encode_f16(repack_scales_to_marlin(
            &group_major,
            fixture.groups,
            fixture.n,
            1,
        ));
        assert_eq!(scales_payload.bytes(), expected_scales.as_ref());
        assert_eq!(scales_payload.element_type(), ElementType::F16);

        let zero_points_component = WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.zero_points").unwrap(),
            role: WeightComponentRole::ZeroPoints,
            external_names: vec![format!("{STEM}.weight_zero_point")],
            dimensions: vec![fixture.groups as u64, (fixture.n / 8) as u64],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::I32,
            },
            required: true,
        };
        let error = source
            .component(&zero_points_component)
            .err()
            .expect("symmetric source must not synthesize zero points");
        assert!(
            error.to_string().contains("absent from safetensors"),
            "{error}"
        );
    }

    #[test]
    fn rejects_drifted_symmetric_metadata_and_zero_point_components() {
        let fixture = write_symmetric_fixture();
        let source =
            CompressedTensorsMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        let component = |quantization| WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                format!("{STEM}.weight_packed"),
                format!("{STEM}.weight_shape"),
            ],
            dimensions: vec![fixture.n as u64, (fixture.k / 2) as u64],
            encoding: WeightEncoding::Quantized(quantization),
            required: true,
        };

        let mut wrong_group = symmetric_quantization();
        wrong_group.grouping = QuantizationGrouping::fixed(64);
        let error = source
            .component(&component(wrong_group))
            .err()
            .expect("wrong symmetric group metadata must be rejected");
        assert!(error.to_string().contains("requires group32"), "{error}");

        let mut fake_zero_point = symmetric_quantization();
        fake_zero_point.zero_point_type = Some(ElementType::I32);
        let error = source
            .component(&component(fake_zero_point))
            .err()
            .expect("symmetric zero-point metadata must be rejected");
        assert!(error.to_string().contains("no zero points"), "{error}");

        let f16_symmetric_fixture = write_fixture_with_options(false, Dtype::F16);
        let f16_symmetric_source =
            CompressedTensorsMarlinSafetensorsSource::open(f16_symmetric_fixture.directory.path())
                .unwrap();
        let scale_component = WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec![format!("{STEM}.weight_scale")],
            dimensions: vec![
                f16_symmetric_fixture.n as u64,
                f16_symmetric_fixture.groups as u64,
            ],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let error = f16_symmetric_source
            .component(&scale_component)
            .err()
            .expect("symmetric F16 source scale must be rejected");
        assert!(
            error
                .to_string()
                .contains("symmetric weight_scale must be BF16"),
            "{error}"
        );

        let f16_asymmetric_fixture = write_fixture_with_options(true, Dtype::F16);
        let f16_asymmetric_source =
            CompressedTensorsMarlinSafetensorsSource::open(f16_asymmetric_fixture.directory.path())
                .unwrap();
        let scale_component = WeightComponentSpec {
            dimensions: vec![
                f16_asymmetric_fixture.n as u64,
                f16_asymmetric_fixture.groups as u64,
            ],
            ..scale_component
        };
        assert!(f16_asymmetric_source.component(&scale_component).is_ok());

        let fixture_with_zero_point = write_fixture();
        let source_with_zero_point = CompressedTensorsMarlinSafetensorsSource::open(
            fixture_with_zero_point.directory.path(),
        )
        .unwrap();
        let packed_component = WeightComponentSpec {
            id: WeightId::new("component.q.symmetric.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                format!("{STEM}.weight_packed"),
                format!("{STEM}.weight_shape"),
            ],
            dimensions: vec![
                fixture_with_zero_point.n as u64,
                (fixture_with_zero_point.k / 2) as u64,
            ],
            encoding: WeightEncoding::Quantized(symmetric_quantization()),
            required: true,
        };
        let error = source_with_zero_point
            .component(&packed_component)
            .err()
            .expect("symmetric physical zero point must be rejected");
        assert!(
            error
                .to_string()
                .contains("must not provide weight_zero_point"),
            "{error}"
        );
    }

    #[test]
    fn rejects_shape_metadata_that_disagrees_with_the_packed_header() {
        let fixture = write_fixture();
        let source =
            CompressedTensorsMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        let component = WeightComponentSpec {
            id: WeightId::new("component.q.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                format!("{STEM}.weight_packed"),
                format!("{STEM}.weight_shape"),
            ],
            dimensions: vec![fixture.n as u64, (fixture.k / 2 + 1) as u64],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        };
        let error = match source.component(&component) {
            Ok(_) => panic!("mismatched typed packed dimensions must be rejected"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("packed component shape"),
            "{error}"
        );
    }
}
