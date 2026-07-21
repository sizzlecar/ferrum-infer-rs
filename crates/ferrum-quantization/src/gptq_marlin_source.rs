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
        let groups = packed_source_groups(component)?;
        let first_qweight = self.tensor(component, groups[0].qweight)?;
        let (k, n) = validate_qweight_shape(component, &first_qweight)?;
        let (expert_count, projections_per_expert) = if groups.len() == 1 {
            let expected_bytes = component.physical_bytes()?;
            if usize::try_from(expected_bytes).ok() != Some(first_qweight.bytes().len()) {
                return Err(invalid_component(
                    component,
                    "qweight byte size differs from the typed packed component",
                ));
            }
            (1, 1)
        } else {
            aggregate_axes(component, groups.len(), n, k / 2, "packed")?
        };
        let fused_n = n.checked_mul(projections_per_expert).ok_or_else(|| {
            invalid_component(component, "fused qweight N dimension exceeds address space")
        })?;
        let expected_bytes = usize::try_from(component.physical_bytes()?).map_err(|_| {
            invalid_component(
                component,
                "packed component byte size exceeds address space",
            )
        })?;
        let mut bytes = Vec::with_capacity(expected_bytes);
        let mut source_files = Vec::with_capacity(component.external_names.len());

        for expert_groups in groups.chunks(projections_per_expert) {
            let mut projections = Vec::with_capacity(projections_per_expert);
            for group in expert_groups {
                let qweight = self.tensor(component, group.qweight)?;
                let shape = validate_qweight_shape(component, &qweight)?;
                if shape != (k, n) {
                    return Err(invalid_component(
                        component,
                        format!(
                            "qweight source `{}` shape K={}, N={} drifts from K={k}, N={n}",
                            group.qweight, shape.0, shape.1
                        ),
                    ));
                }
                let qzeros = self.tensor(component, group.qzeros)?;
                validate_symmetric_qzeros_shape(
                    component,
                    &qzeros,
                    k,
                    n,
                    quantization.group_size as usize,
                )?;
                source_files.push(qweight.source_file().to_owned());
                source_files.push(qzeros.source_file().to_owned());
                if let Some(g_idx_name) = group.g_idx {
                    let g_idx = self.tensor(component, g_idx_name)?;
                    validate_canonical_g_idx(
                        component,
                        &g_idx,
                        k,
                        quantization.group_size as usize,
                    )?;
                    source_files.push(g_idx.source_file().to_owned());
                }
                projections.push(decode_i32(qweight.bytes(), component, "qweight")?);
            }
            let fused = concatenate_equal_width_rows(&projections, k / 8, n);
            let repacked = repack_gptq_to_marlin(&fused, k, fused_n);
            bytes.extend_from_slice(encode_i32(repacked).as_ref());
        }
        debug_assert_eq!(groups.len(), expert_count * projections_per_expert);
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
        if component.external_names.is_empty() {
            return Err(invalid_component(
                component,
                "Marlin scales require at least one safetensors source",
            ));
        }
        if component
            .external_names
            .iter()
            .any(|external_name| !external_name.ends_with(".scales"))
        {
            return Err(invalid_component(
                component,
                "every Marlin scale source must end with .scales",
            ));
        }
        let external_name = &component.external_names[0];
        let scales = self.tensor(component, external_name)?;
        let (group_count, n) = validate_scale_shape(component, &scales)?;
        let (expert_count, projections_per_expert) = if component.external_names.len() == 1 {
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
            (1, 1)
        } else {
            aggregate_axes(
                component,
                component.external_names.len(),
                n,
                group_count,
                "scale",
            )?
        };
        let fused_n = n.checked_mul(projections_per_expert).ok_or_else(|| {
            invalid_component(component, "fused scale N dimension exceeds address space")
        })?;
        let expected_bytes = usize::try_from(component.physical_bytes()?).map_err(|_| {
            invalid_component(component, "scale component byte size exceeds address space")
        })?;
        let mut bytes = Vec::with_capacity(expected_bytes);
        let mut source_files = Vec::with_capacity(component.external_names.len());

        for expert_names in component.external_names.chunks(projections_per_expert) {
            let mut projections = Vec::with_capacity(projections_per_expert);
            for external_name in expert_names {
                let scales = self.tensor(component, external_name)?;
                let shape = validate_scale_shape(component, &scales)?;
                if shape != (group_count, n) {
                    return Err(invalid_component(
                        component,
                        format!(
                            "scale source `{external_name}` shape [{}, {}] drifts from [{group_count}, {n}]",
                            shape.0, shape.1
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
                projections.push(decode_f16(&f16_bytes, component)?);
                source_files.push(scales.source_file().to_owned());
            }
            let fused = concatenate_equal_width_rows(&projections, group_count, n);
            let repacked = repack_scales_to_marlin(&fused, group_count, fused_n, 1);
            bytes.extend_from_slice(encode_f16(repacked).as_ref());
        }
        debug_assert_eq!(
            component.external_names.len(),
            expert_count * projections_per_expert
        );
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            source_files,
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

#[derive(Clone, Copy)]
struct PackedSourceGroup<'name> {
    qweight: &'name str,
    qzeros: &'name str,
    g_idx: Option<&'name str>,
}

fn packed_source_groups(
    component: &WeightComponentSpec,
) -> std::result::Result<Vec<PackedSourceGroup<'_>>, VNextError> {
    if component.external_names.is_empty() {
        return Err(invalid_component(
            component,
            "packed GPTQ values require ordered qweight and qzeros sources",
        ));
    }
    let mut groups = Vec::new();
    let mut cursor = 0;
    let mut expected_g_idx_presence = None;
    while cursor < component.external_names.len() {
        let qweight = &component.external_names[cursor];
        let stem = qweight.strip_suffix(".qweight").unwrap_or_default();
        let Some(qzeros) = component.external_names.get(cursor + 1) else {
            return Err(invalid_component(
                component,
                "each packed GPTQ source group requires qweight followed by qzeros",
            ));
        };
        if stem.is_empty() || qzeros != &format!("{stem}.qzeros") {
            return Err(invalid_component(
                component,
                "packed GPTQ source groups must share one stem and be ordered qweight, qzeros, then optional g_idx",
            ));
        }
        let expected_g_idx = format!("{stem}.g_idx");
        let g_idx = component
            .external_names
            .get(cursor + 2)
            .filter(|name| name.as_str() == expected_g_idx)
            .map(String::as_str);
        let has_g_idx = g_idx.is_some();
        if expected_g_idx_presence
            .replace(has_g_idx)
            .is_some_and(|expected| expected != has_g_idx)
        {
            return Err(invalid_component(
                component,
                "packed GPTQ source groups cannot mix g_idx presence",
            ));
        }
        groups.push(PackedSourceGroup {
            qweight,
            qzeros,
            g_idx,
        });
        cursor += if has_g_idx { 3 } else { 2 };
    }
    Ok(groups)
}

fn aggregate_axes(
    component: &WeightComponentSpec,
    source_group_count: usize,
    source_n: usize,
    source_tail: usize,
    label: &str,
) -> std::result::Result<(usize, usize), VNextError> {
    let [expert_count, projections_per_expert, typed_n, typed_tail] =
        component.dimensions.as_slice()
    else {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} shape must be [E, P, N, physical_K], got {:?}",
                component.dimensions
            ),
        ));
    };
    let expected_tail = [source_n as u64, source_tail as u64];
    if [*typed_n, *typed_tail] != expected_tail {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} tail [{typed_n}, {typed_tail}] must match single-source physical shape {expected_tail:?}"
            ),
        ));
    }
    let declared_groups = expert_count
        .checked_mul(*projections_per_expert)
        .ok_or_else(|| {
            invalid_component(component, "aggregate source group count overflows u64")
        })?;
    if *expert_count == 0
        || *projections_per_expert == 0
        || usize::try_from(declared_groups).ok() != Some(source_group_count)
    {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} prefix E={expert_count}, P={projections_per_expert} must describe {source_group_count} ordered source groups"
            ),
        ));
    }
    Ok((
        usize::try_from(*expert_count).map_err(|_| {
            invalid_component(component, "aggregate expert count exceeds address space")
        })?,
        usize::try_from(*projections_per_expert).map_err(|_| {
            invalid_component(
                component,
                "aggregate projection count exceeds address space",
            )
        })?,
    ))
}

fn validate_qweight_shape(
    component: &WeightComponentSpec,
    qweight: &SafetensorsTensor<'_>,
) -> std::result::Result<(usize, usize), VNextError> {
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
    if k % 16 != 0 || n % 16 != 0 || k.checked_mul(n).is_none_or(|elements| elements % 1024 != 0) {
        return Err(invalid_component(
            component,
            format!("qweight shape K={k}, N={n} is not Marlin tile aligned"),
        ));
    }
    Ok((k, n))
}

fn validate_scale_shape(
    component: &WeightComponentSpec,
    scales: &SafetensorsTensor<'_>,
) -> std::result::Result<(usize, usize), VNextError> {
    let [group_count, n] = scales.shape() else {
        return Err(invalid_component(
            component,
            format!(
                "scales must have source shape [K/G, N], got {:?}",
                scales.shape()
            ),
        ));
    };
    Ok((
        usize::try_from(*group_count)
            .map_err(|_| invalid_component(component, "scale group count exceeds address space"))?,
        usize::try_from(*n)
            .map_err(|_| invalid_component(component, "scale N dimension exceeds address space"))?,
    ))
}

fn concatenate_equal_width_rows<T: Copy>(
    parts: &[Vec<T>],
    row_count: usize,
    columns_per_part: usize,
) -> Vec<T> {
    let mut fused = Vec::with_capacity(row_count * columns_per_part * parts.len());
    for row in 0..row_count {
        for part in parts {
            let start = row * columns_per_part;
            fused.extend_from_slice(&part[start..start + columns_per_part]);
        }
    }
    fused
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

    struct GateUpFixture {
        directory: tempfile::TempDir,
        qweights: [Vec<i32>; 2],
        scales: [Vec<f16>; 2],
        k: usize,
        n: usize,
    }

    fn write_gate_up_fixture() -> GateUpFixture {
        let directory = tempdir().unwrap();
        let k = 128_usize;
        let n = 16_usize;
        let qweights = [
            (0..(k / 8) * n)
                .map(|index| (index as u32).wrapping_mul(0x1020_4081) as i32)
                .collect::<Vec<_>>(),
            (0..(k / 8) * n)
                .map(|index| {
                    (index as u32)
                        .wrapping_mul(0x0810_2041)
                        .wrapping_add(0x7654_3210) as i32
                })
                .collect::<Vec<_>>(),
        ];
        let scales = [
            (0..n)
                .map(|index| f16::from_f32(index as f32 + 1.0))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|index| f16::from_f32(index as f32 + 101.0))
                .collect::<Vec<_>>(),
        ];
        let qweight_bytes = qweights.each_ref().map(|values| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        });
        let scale_bytes = scales.each_ref().map(|values| {
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        });
        let qzeros = vec![0x8888_8888_u32 as i32; n / 8];
        let qzeros_bytes = qzeros
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let g_idx = vec![0_i32; k];
        let g_idx_bytes = g_idx
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let views = BTreeMap::from([
            (
                "layer.gate.g_idx",
                TensorView::new(Dtype::I32, vec![k], &g_idx_bytes).unwrap(),
            ),
            (
                "layer.gate.qweight",
                TensorView::new(Dtype::I32, vec![k / 8, n], &qweight_bytes[0]).unwrap(),
            ),
            (
                "layer.gate.qzeros",
                TensorView::new(Dtype::I32, vec![1, n / 8], &qzeros_bytes).unwrap(),
            ),
            (
                "layer.gate.scales",
                TensorView::new(Dtype::F16, vec![1, n], &scale_bytes[0]).unwrap(),
            ),
            (
                "layer.up.g_idx",
                TensorView::new(Dtype::I32, vec![k], &g_idx_bytes).unwrap(),
            ),
            (
                "layer.up.qweight",
                TensorView::new(Dtype::I32, vec![k / 8, n], &qweight_bytes[1]).unwrap(),
            ),
            (
                "layer.up.qzeros",
                TensorView::new(Dtype::I32, vec![1, n / 8], &qzeros_bytes).unwrap(),
            ),
            (
                "layer.up.scales",
                TensorView::new(Dtype::F16, vec![1, n], &scale_bytes[1]).unwrap(),
            ),
        ]);
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        GateUpFixture {
            directory,
            qweights,
            scales,
            k,
            n,
        }
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

    #[test]
    fn aggregate_gate_up_fuses_raw_columns_before_marlin_repack() {
        let fixture = write_gate_up_fixture();
        let source = GptqMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        let packed = WeightComponentSpec {
            id: WeightId::new("component.layer.gate_up.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                "layer.gate.qweight".to_owned(),
                "layer.gate.qzeros".to_owned(),
                "layer.gate.g_idx".to_owned(),
                "layer.up.qweight".to_owned(),
                "layer.up.qzeros".to_owned(),
                "layer.up.g_idx".to_owned(),
            ],
            dimensions: vec![1, 2, fixture.n as u64, (fixture.k / 2) as u64],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        };
        let raw_fused = concatenate_equal_width_rows(&fixture.qweights, fixture.k / 8, fixture.n);
        let expected = encode_i32(repack_gptq_to_marlin(&raw_fused, fixture.k, fixture.n * 2));
        let independently_repacked = fixture
            .qweights
            .iter()
            .flat_map(|values| {
                repack_gptq_to_marlin(values, fixture.k, fixture.n)
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        assert_ne!(expected.as_ref(), independently_repacked);
        let payload = source.component(&packed).unwrap();
        assert_eq!(payload.bytes(), expected.as_ref());
        assert_eq!(payload.external_names(), packed.external_names);

        let scales = WeightComponentSpec {
            id: WeightId::new("component.layer.gate_up.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec!["layer.gate.scales".to_owned(), "layer.up.scales".to_owned()],
            dimensions: vec![1, 2, fixture.n as u64, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let raw_fused_scales = concatenate_equal_width_rows(&fixture.scales, 1, fixture.n);
        let expected_scales = encode_f16(repack_scales_to_marlin(
            &raw_fused_scales,
            1,
            fixture.n * 2,
            1,
        ));
        let independently_repacked_scales = fixture
            .scales
            .iter()
            .flat_map(|values| {
                repack_scales_to_marlin(values, 1, fixture.n, 1)
                    .into_iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
            })
            .collect::<Vec<_>>();
        assert_ne!(expected_scales.as_ref(), independently_repacked_scales);
        let payload = source.component(&scales).unwrap();
        assert_eq!(payload.bytes(), expected_scales.as_ref());
        assert_eq!(payload.external_names(), scales.external_names);
    }
}
