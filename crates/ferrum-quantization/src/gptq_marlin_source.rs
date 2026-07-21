//! Typed safetensors GPTQ adapter for the Marlin physical ABI.
//!
//! The adapter performs validation and repacking once while static plan
//! resources are initialized. Providers receive only plan-owned device
//! regions and never parse checkpoint metadata or repack on a request path.

use std::borrow::Cow;
use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, QuantizationSpec, VNextError, WeightComponentPayload, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_kernels::marlin_repack::{repack_gptq_to_marlin, repack_scales_to_marlin};
use ferrum_types::Result;
use half::f16;
use safetensors::Dtype;

use crate::safetensors_archive::{transcode_dense_bytes, SafetensorsArchive, SafetensorsTensor};

pub const GPTQ_MARLIN_INT4_FORMAT_ID: &str = "quantization.marlin.gptq-int4-symmetric";

/// Mmap-backed safetensors archive with an explicit GPTQ-to-Marlin cold-path
/// adapter. Dense components retain the archive's zero-copy behavior.
pub struct GptqMarlinSafetensorsSource {
    archive: SafetensorsArchive,
}

impl GptqMarlinSafetensorsSource {
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
        validate_marlin_quantization(component, quantization)?;
        let Some(qweight_name) = component.external_names.first() else {
            return Err(invalid_component(
                component,
                "packed GPTQ values require ordered qweight and qzeros sources",
            ));
        };
        let Some(qzeros_name) = component.external_names.get(1) else {
            return Err(invalid_component(
                component,
                "packed GPTQ values require ordered qweight and qzeros sources",
            ));
        };
        let qweight_stem = qweight_name.strip_suffix(".qweight").unwrap_or_default();
        if qweight_stem.is_empty()
            || qzeros_name != &format!("{qweight_stem}.qzeros")
            || component
                .external_names
                .get(2)
                .is_some_and(|name| name != &format!("{qweight_stem}.g_idx"))
        {
            return Err(invalid_component(
                component,
                "packed GPTQ sources must share one stem and be ordered qweight, qzeros, then optional g_idx",
            ));
        }
        if component.external_names.len() > 3
            || component
                .external_names
                .get(2)
                .is_some_and(|name| !name.ends_with(".g_idx"))
        {
            return Err(invalid_component(
                component,
                "packed GPTQ source has an unsupported sidecar set",
            ));
        }

        let qweight = self.tensor(component, qweight_name)?;
        if qweight.dtype() != Dtype::I32 {
            return Err(invalid_component(
                component,
                format!("qweight must be I32, got {:?}", qweight.dtype()),
            ));
        }
        let [packed_k, n] = qweight.shape() else {
            return Err(invalid_component(
                component,
                format!(
                    "qweight must have shape [K/8, N], got {:?}",
                    qweight.shape()
                ),
            ));
        };
        let k = packed_k.checked_mul(8).ok_or_else(|| {
            invalid_component(component, "qweight K dimension overflows address space")
        })?;
        let (k, n) = (
            usize::try_from(k).map_err(|_| {
                invalid_component(component, "qweight K dimension exceeds address space")
            })?,
            usize::try_from(*n).map_err(|_| {
                invalid_component(component, "qweight N dimension exceeds address space")
            })?,
        );
        if k % 16 != 0 || n % 16 != 0 || (k * n) % 1024 != 0 {
            return Err(invalid_component(
                component,
                format!("qweight shape K={k}, N={n} is not Marlin tile aligned"),
            ));
        }
        let expected_bytes = component.physical_bytes()?;
        if usize::try_from(expected_bytes).ok() != Some(qweight.bytes().len()) {
            return Err(invalid_component(
                component,
                "qweight byte size differs from the typed packed component",
            ));
        }

        let qzeros = self.tensor(component, qzeros_name)?;
        validate_symmetric_qzeros_shape(
            component,
            &qzeros,
            k,
            n,
            quantization.group_size as usize,
        )?;
        if let Some(g_idx_name) = component.external_names.get(2) {
            let g_idx = self.tensor(component, g_idx_name)?;
            validate_canonical_g_idx(component, &g_idx, k, quantization.group_size as usize)?;
        }

        let qweight_words = decode_i32(qweight.bytes(), component, "qweight")?;
        let repacked = repack_gptq_to_marlin(&qweight_words, k, n);
        let bytes = encode_i32(repacked);
        let mut source_files = vec![
            qweight.source_file().to_owned(),
            qzeros.source_file().to_owned(),
        ];
        if let Some(g_idx_name) = component.external_names.get(2) {
            source_files.push(self.tensor(component, g_idx_name)?.source_file().to_owned());
        }
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            source_files,
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
                "Marlin scales require exactly one safetensors source",
            ));
        };
        if !external_name.ends_with(".scales") {
            return Err(invalid_component(
                component,
                "Marlin scale source must end with .scales",
            ));
        }
        let scales = self.tensor(component, external_name)?;
        let [group_count, n] = scales.shape() else {
            return Err(invalid_component(
                component,
                format!(
                    "scales must have source shape [K/G, N], got {:?}",
                    scales.shape()
                ),
            ));
        };
        let (group_count, n) = (
            usize::try_from(*group_count).map_err(|_| {
                invalid_component(component, "scale group count exceeds address space")
            })?,
            usize::try_from(*n).map_err(|_| {
                invalid_component(component, "scale N dimension exceeds address space")
            })?,
        );
        let mut expected_dimensions = vec![1_u64; component.dimensions.len().saturating_sub(2)];
        expected_dimensions.extend([n as u64, group_count as u64]);
        if component.dimensions != expected_dimensions {
            return Err(invalid_component(
                component,
                format!(
                    "typed scale shape {:?} must be {:?} for source shape [{group_count}, {n}]",
                    component.dimensions, expected_dimensions,
                ),
            ));
        }
        let source_type = scales.element_type().ok_or_else(|| {
            invalid_component(
                component,
                format!("scales have unsupported dtype {:?}", scales.dtype()),
            )
        })?;
        let f16_bytes = transcode_dense_bytes(
            scales.bytes(),
            source_type,
            ElementType::F16,
            external_name,
            None,
        )?;
        let values = decode_f16(&f16_bytes, component)?;
        let repacked = repack_scales_to_marlin(&values, group_count, n, 1);
        let bytes = encode_f16(repacked);
        WeightComponentPayload::new(
            component,
            scales.external_name(),
            scales.source_file(),
            component.dimensions.clone(),
            ElementType::F16,
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

impl WeightComponentSource for GptqMarlinSafetensorsSource {
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
            (_, WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. }) => {
                self.archive.component(component)
            }
            _ => Err(invalid_component(
                component,
                "GPTQ Marlin adapter received an unsupported component encoding",
            )),
        }
    }
}

fn validate_marlin_quantization(
    component: &WeightComponentSpec,
    quantization: &QuantizationSpec,
) -> std::result::Result<(), VNextError> {
    quantization.validate()?;
    if quantization.format_id.as_str() != GPTQ_MARLIN_INT4_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.group_size == 0
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type.is_some()
    {
        return Err(invalid_component(
            component,
            "typed GPTQ source requires symmetric INT4 Marlin packing with F16 scales",
        ));
    }
    Ok(())
}

fn validate_symmetric_qzeros_shape(
    component: &WeightComponentSpec,
    qzeros: &SafetensorsTensor<'_>,
    k: usize,
    n: usize,
    group_size: usize,
) -> std::result::Result<(), VNextError> {
    if qzeros.dtype() != Dtype::I32
        || group_size == 0
        || qzeros.shape() != [k as u64 / group_size as u64, n as u64 / 8]
    {
        return Err(invalid_component(
            component,
            format!(
                "qzeros shape/dtype differs from symmetric GPTQ K={k}, N={n}, group_size={group_size}"
            ),
        ));
    }
    // `sym=true` selects Marlin's fixed uint4b8 bias. GPTQ writers use more
    // than one historical qzeros convention even though the sidecar is not
    // consumed for symmetric inference, so its contents must not define the
    // physical ABI. Identity, dtype, and shape remain strict.
    Ok(())
}

fn validate_canonical_g_idx(
    component: &WeightComponentSpec,
    g_idx: &SafetensorsTensor<'_>,
    k: usize,
    group_size: usize,
) -> std::result::Result<(), VNextError> {
    if g_idx.dtype() != Dtype::I32 || g_idx.shape() != [k as u64] {
        return Err(invalid_component(
            component,
            format!("g_idx must be I32[{k}] for desc_act=false"),
        ));
    }
    let values = decode_i32(g_idx.bytes(), component, "g_idx")?;
    if values
        .iter()
        .enumerate()
        .any(|(index, value)| *value != (index / group_size) as i32)
    {
        return Err(invalid_component(
            component,
            "g_idx is activation-ordered; the current typed Marlin ABI requires desc_act=false",
        ));
    }
    Ok(())
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

fn encode_i32(values: Vec<i32>) -> Cow<'static, [u8]> {
    Cow::Owned(
        values
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
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
        .map(|word| f16::from_bits(u16::from_le_bytes([word[0], word[1]])))
        .collect())
}

fn encode_f16(values: Vec<f16>) -> Cow<'static, [u8]> {
    Cow::Owned(
        values
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn invalid_component(component: &WeightComponentSpec, reason: impl AsRef<str>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!(
            "GPTQ Marlin component `{}`: {}",
            component.id,
            reason.as_ref()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ferrum_interfaces::vnext::{QuantizationFormatId, QuantizationPacking, WeightId};
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    fn write_fixture(qzeros_word: i32) -> tempfile::TempDir {
        let directory = tempdir().unwrap();
        let k = 128_usize;
        let n = 64_usize;
        let qweight_words = vec![0x7654_3210_i32; (k / 8) * n];
        let qzeros_words = vec![qzeros_word; n / 8];
        let g_idx = (0..k).map(|_| 0_i32).collect::<Vec<_>>();
        let scales = vec![f16::from_f32(0.5); n];
        let qweight_bytes = qweight_words
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let qzeros_bytes = qzeros_words
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let g_idx_bytes = g_idx
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let scale_bytes = scales
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let views = BTreeMap::from([
            (
                "layer.proj.g_idx",
                TensorView::new(Dtype::I32, vec![k], &g_idx_bytes).unwrap(),
            ),
            (
                "layer.proj.qweight",
                TensorView::new(Dtype::I32, vec![k / 8, n], &qweight_bytes).unwrap(),
            ),
            (
                "layer.proj.qzeros",
                TensorView::new(Dtype::I32, vec![1, n / 8], &qzeros_bytes).unwrap(),
            ),
            (
                "layer.proj.scales",
                TensorView::new(Dtype::F16, vec![1, n], &scale_bytes).unwrap(),
            ),
        ]);
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(GPTQ_MARLIN_INT4_FORMAT_ID).unwrap(),
            bits_per_weight: 4,
            group_size: 128,
            packing: QuantizationPacking::Tiled,
            scale_type: ElementType::F16,
            zero_point_type: None,
        }
    }

    fn packed_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.layer.proj.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                "layer.proj.qweight".to_owned(),
                "layer.proj.qzeros".to_owned(),
                "layer.proj.g_idx".to_owned(),
            ],
            dimensions: vec![4096],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        }
    }

    #[test]
    fn repacks_valid_symmetric_gptq_components_once_at_source_boundary() {
        let directory = write_fixture(0x8888_8888_u32 as i32);
        let source = GptqMarlinSafetensorsSource::open(directory.path()).unwrap();
        let packed = packed_component();
        let payload = source.component(&packed).unwrap();
        assert_eq!(payload.bytes().len(), 4096);
        assert_eq!(payload.external_names(), packed.external_names);

        let scales = WeightComponentSpec {
            id: WeightId::new("component.layer.proj.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec!["layer.proj.scales".to_owned()],
            dimensions: vec![64, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let payload = source.component(&scales).unwrap();
        assert_eq!(payload.bytes().len(), 128);
        assert_eq!(payload.dimensions(), [64, 1]);
    }

    #[test]
    fn symmetric_qzeros_convention_does_not_change_marlin_payload() {
        let code7 = write_fixture(0x7777_7777);
        let code8 = write_fixture(0x8888_8888_u32 as i32);
        let source7 = GptqMarlinSafetensorsSource::open(code7.path()).unwrap();
        let source8 = GptqMarlinSafetensorsSource::open(code8.path()).unwrap();
        let component = packed_component();

        assert_eq!(
            source7.component(&component).unwrap().bytes(),
            source8.component(&component).unwrap().bytes()
        );
    }
}
