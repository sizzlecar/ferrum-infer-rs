//! Typed safetensors GPTQ adapter for the Marlin physical ABI.
//!
//! The adapter performs validation and repacking once while static plan
//! resources are initialized. Providers receive only plan-owned device
//! regions and never parse checkpoint metadata or repack on a request path.

use std::borrow::Cow;
use std::path::Path;

#[cfg(test)]
use ferrum_interfaces::vnext::QuantizationGrouping;
use ferrum_interfaces::vnext::{
    ElementType, QuantizationSpec, VNextError, WeightComponentPayload, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
#[cfg(test)]
use ferrum_kernels::marlin_repack::repack_gptq_to_marlin;
use ferrum_kernels::marlin_repack::{repack_gptq_to_marlin_bytes_into, repack_scales_to_marlin};
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

    /// Materialize a small symmetric GPTQ matrix as row-major F16. The schema
    /// explicitly names all four sources in qweight, qzeros, g_idx, scales
    /// order; ordinary dense tensors and native Marlin matrices keep their
    /// existing paths. This happens only during static resource initialization.
    fn dense_gptq<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [qweight_name, qzeros_name, g_idx_name, scales_name] =
            component.external_names.as_slice()
        else {
            return Err(invalid_component(
                component,
                "dense GPTQ requires four ordered sources",
            ));
        };
        let stem = qweight_name.strip_suffix(".qweight").unwrap_or_default();
        if stem.is_empty()
            || qzeros_name != &format!("{stem}.qzeros")
            || g_idx_name != &format!("{stem}.g_idx")
            || scales_name != &format!("{stem}.scales")
        {
            return Err(invalid_component(component, "dense GPTQ sources must share one stem and be ordered qweight, qzeros, g_idx, scales"));
        }
        let qweight = self.tensor(component, qweight_name)?;
        let qzeros = self.tensor(component, qzeros_name)?;
        let g_idx = self.tensor(component, g_idx_name)?;
        let scales = self.tensor(component, scales_name)?;
        let [n, k] = component.dimensions.as_slice() else {
            return Err(invalid_component(
                component,
                "dense GPTQ output must be a matrix [N, K]",
            ));
        };
        let n = usize::try_from(*n)
            .map_err(|_| invalid_component(component, "dense GPTQ N overflows"))?;
        let k = usize::try_from(*k)
            .map_err(|_| invalid_component(component, "dense GPTQ K overflows"))?;
        let (group_count, scale_n) = validate_scale_shape(component, &scales)?;
        if n == 0
            || k == 0
            || !n.is_multiple_of(8)
            || !k.is_multiple_of(8)
            || group_count == 0
            || !k.is_multiple_of(group_count)
            || scale_n != n
            || qweight.dtype() != Dtype::I32
            || qweight.shape() != [k as u64 / 8, n as u64]
            || scales.dtype() != Dtype::F16
        {
            return Err(invalid_component(
                component,
                "dense GPTQ source shape/dtype differs from its F16 matrix contract",
            ));
        }
        let group_size = k / group_count;
        validate_symmetric_qzeros_shape(component, &qzeros, k, n, group_size)?;
        validate_canonical_g_idx(component, &g_idx, k, group_size)?;
        let packed = decode_i32(qweight.bytes(), component, "qweight")?;
        let scales_values = decode_f16(scales.bytes(), component)?;
        let byte_count = usize::try_from(component.physical_bytes()?)
            .map_err(|_| invalid_component(component, "dense GPTQ byte count overflows"))?;
        let mut bytes = Vec::with_capacity(byte_count);
        for output in 0..n {
            for input in 0..k {
                let word = packed[(input / 8) * n + output] as u32;
                let code = ((word >> ((input % 8) * 4)) & 15) as i32;
                // The adapter's symmetric INT4 contract uses the same uint4b8
                // bias as Marlin, independently of historical qzeros encoding.
                let value =
                    (code - 8) as f32 * scales_values[(input / group_size) * n + output].to_f32();
                let value = f16::from_f32(value);
                if !value.is_finite() {
                    return Err(invalid_component(
                        component,
                        "dense GPTQ produced a non-finite F16 weight",
                    ));
                }
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            [&qweight, &qzeros, &g_idx, &scales]
                .map(|tensor| tensor.source_file().to_owned())
                .to_vec(),
            component.dimensions.clone(),
            ElementType::F16,
            bytes,
        )
    }

    fn packed_values<'source>(
        &'source self,
        component: &WeightComponentSpec,
        quantization: &QuantizationSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let group_size = usize::try_from(validate_marlin_quantization(component, quantization)?)
            .map_err(|_| invalid_component(component, "GPTQ group size exceeds address space"))?;
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
                validate_symmetric_qzeros_shape(component, &qzeros, k, n, group_size)?;
                source_files.push(qweight.source_file().to_owned());
                source_files.push(qzeros.source_file().to_owned());
                if let Some(g_idx_name) = group.g_idx {
                    let g_idx = self.tensor(component, g_idx_name)?;
                    validate_canonical_g_idx(component, &g_idx, k, group_size)?;
                    source_files.push(g_idx.source_file().to_owned());
                }
                projections.push(decode_i32(qweight.bytes(), component, "qweight")?);
            }
            let fused = concatenate_equal_width_rows(&projections, k / 8, n);
            let start = bytes.len();
            let byte_length = fused
                .len()
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| {
                    invalid_component(component, "repacked qweight byte length overflows")
                })?;
            let end = start.checked_add(byte_length).ok_or_else(|| {
                invalid_component(component, "aggregate qweight byte length overflows")
            })?;
            bytes.resize(end, 0);
            repack_gptq_to_marlin_bytes_into(&fused, k, fused_n, &mut bytes[start..end]);
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
            (
                WeightComponentRole::Values,
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
            ) if component
                .external_names
                .first()
                .is_some_and(|name| name.ends_with(".qweight")) =>
            {
                self.dense_gptq(component)
            }
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
    if component.dimensions.len() < 3 {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} shape must be [E, projection_axes..., N, physical_K], got {:?}",
                component.dimensions
            ),
        ));
    }
    let tail_start = component.dimensions.len() - 2;
    let expert_count = component.dimensions[0];
    let typed_n = component.dimensions[tail_start];
    let typed_tail = component.dimensions[tail_start + 1];
    let expected_tail = [source_n as u64, source_tail as u64];
    if [typed_n, typed_tail] != expected_tail {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} tail [{typed_n}, {typed_tail}] must match single-source physical shape {expected_tail:?}"
            ),
        ));
    }
    let projections_per_expert = component.dimensions[1..tail_start]
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| {
            invalid_component(component, "aggregate projection axis product overflows u64")
        })?;
    let declared_groups = expert_count
        .checked_mul(projections_per_expert)
        .ok_or_else(|| {
            invalid_component(component, "aggregate source group count overflows u64")
        })?;
    if expert_count == 0
        || projections_per_expert == 0
        || usize::try_from(declared_groups).ok() != Some(source_group_count)
    {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} prefix E={expert_count}, projections_per_expert={projections_per_expert} must describe {source_group_count} ordered source groups"
            ),
        ));
    }
    Ok((
        usize::try_from(expert_count).map_err(|_| {
            invalid_component(component, "aggregate expert count exceeds address space")
        })?,
        usize::try_from(projections_per_expert).map_err(|_| {
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
) -> std::result::Result<u32, VNextError> {
    quantization.validate()?;
    let Some(group_size) = quantization.grouping.fixed_size() else {
        return Err(invalid_component(
            component,
            "typed GPTQ source requires fixed-size quantization groups",
        ));
    };
    if quantization.format_id.as_str() != GPTQ_MARLIN_INT4_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type.is_some()
    {
        return Err(invalid_component(
            component,
            "typed GPTQ source requires symmetric INT4 Marlin packing with F16 scales",
        ));
    }
    Ok(group_size)
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
            grouping: QuantizationGrouping::fixed(128),
            packing: QuantizationPacking::Tiled,
            scale_type: ElementType::F16,
            zero_point_type: None,
        }
    }

    type OwnedTensor = (Dtype, Vec<usize>, Vec<u8>);

    fn dense_fixture(
        qzeros_word: u32,
        mutate: impl FnOnce(&mut BTreeMap<String, OwnedTensor>),
    ) -> tempfile::TempDir {
        let (n, k, group_size) = (8_usize, 256_usize, 128_usize);
        let mut words = vec![0_u32; k / 8 * n];
        for input in 0..k {
            for output in 0..n {
                words[input / 8 * n + output] |=
                    (((input + 3 * output) % 16) as u32) << (4 * (input % 8));
            }
        }
        let mut tensors = BTreeMap::from([
            (
                "layer.proj.qweight".to_owned(),
                (
                    Dtype::I32,
                    vec![k / 8, n],
                    words.into_iter().flat_map(u32::to_le_bytes).collect(),
                ),
            ),
            (
                "layer.proj.qzeros".to_owned(),
                (
                    Dtype::I32,
                    vec![k / group_size, n / 8],
                    (0..k / group_size * n / 8)
                        .flat_map(|_| qzeros_word.to_le_bytes())
                        .collect(),
                ),
            ),
            (
                "layer.proj.g_idx".to_owned(),
                (
                    Dtype::I32,
                    vec![k],
                    (0..k)
                        .flat_map(|input| ((input / group_size) as i32).to_le_bytes())
                        .collect(),
                ),
            ),
            (
                "layer.proj.scales".to_owned(),
                (
                    Dtype::F16,
                    vec![k / group_size, n],
                    (0..k / group_size)
                        .flat_map(|group| {
                            (0..n).flat_map(move |output| {
                                f16::from_f32((1 + group * 2 + output) as f32 * 0.25)
                                    .to_bits()
                                    .to_le_bytes()
                            })
                        })
                        .collect(),
                ),
            ),
        ]);
        mutate(&mut tensors);
        let directory = tempdir().unwrap();
        let views = tensors
            .iter()
            .map(|(name, (dtype, shape, bytes))| {
                (name, TensorView::new(*dtype, shape.clone(), bytes).unwrap())
            })
            .collect::<BTreeMap<_, _>>();
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn dense_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.layer.proj.values").unwrap(),
            role: WeightComponentRole::Values,
            external_names: ["qweight", "qzeros", "g_idx", "scales"]
                .map(|suffix| format!("layer.proj.{suffix}"))
                .to_vec(),
            dimensions: vec![8, 256],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }
    }

    #[test]
    fn dense_symmetric_gptq_preserves_rows_groups_signs_and_source_identity() {
        let component = dense_component();
        for qzeros in [0x7777_7777, 0x8888_8888] {
            let directory = dense_fixture(qzeros, |_| {});
            let source = GptqMarlinSafetensorsSource::open(directory.path()).unwrap();
            let payload = source.component(&component).unwrap();
            assert_eq!(payload.dimensions(), [8, 256]);
            assert_eq!(payload.external_names(), component.external_names);
            let values = decode_f16(payload.bytes(), &component).unwrap();
            for output in 0..8 {
                for input in 0..256 {
                    let code = ((input + 3 * output) % 16) as i32;
                    let scale = (1 + (input / 128) * 2 + output) as f32 * 0.25;
                    assert_eq!(
                        values[output * 256 + input].to_f32(),
                        (code - 8) as f32 * scale
                    );
                }
            }
        }
    }

    #[test]
    fn dense_gptq_rejects_invalid_source_recipes_and_dimensions() {
        let directory = dense_fixture(0x7777_7777, |_| {});
        let source = GptqMarlinSafetensorsSource::open(directory.path()).unwrap();
        let component = dense_component();
        let mut invalid = vec![];
        let mut wrong_order = component.clone();
        wrong_order.external_names.swap(1, 2);
        invalid.push(wrong_order);
        let mut wrong_stem = component.clone();
        wrong_stem.external_names[3] = "other.scales".into();
        invalid.push(wrong_stem);
        let mut missing = component.clone();
        missing.external_names.pop();
        invalid.push(missing);
        for dimensions in [
            vec![256, 8],
            vec![4, 512],
            vec![0, 256],
            vec![8, 128],
            vec![2, 4, 256],
            vec![8, u64::MAX],
        ] {
            let mut wrong_shape = component.clone();
            wrong_shape.dimensions = dimensions;
            invalid.push(wrong_shape);
        }
        for invalid in invalid {
            assert!(source.component(&invalid).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn dense_gptq_rejects_bad_payloads_before_materialization() {
        for case in 0..7 {
            let directory = dense_fixture(0x7777_7777, |tensors| match case {
                0 => tensors.get_mut("layer.proj.g_idx").unwrap().2[..4]
                    .copy_from_slice(&1_i32.to_le_bytes()),
                1 => tensors.get_mut("layer.proj.scales").unwrap().0 = Dtype::BF16,
                2 => tensors.get_mut("layer.proj.scales").unwrap().1 = vec![1, 16],
                3 => tensors.get_mut("layer.proj.qzeros").unwrap().0 = Dtype::F32,
                4 => {
                    tensors.remove("layer.proj.g_idx");
                }
                5 => tensors.get_mut("layer.proj.scales").unwrap().2[..2]
                    .copy_from_slice(&f16::NAN.to_bits().to_le_bytes()),
                6 => tensors.get_mut("layer.proj.scales").unwrap().2[..2]
                    .copy_from_slice(&f16::MAX.to_bits().to_le_bytes()),
                _ => unreachable!(),
            });
            let source = GptqMarlinSafetensorsSource::open(directory.path()).unwrap();
            assert!(source.component(&dense_component()).is_err(), "case {case}");
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
        let expected = repack_gptq_to_marlin(&raw_fused, fixture.k, fixture.n * 2)
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let independently_repacked = fixture
            .qweights
            .iter()
            .flat_map(|values| {
                repack_gptq_to_marlin(values, fixture.k, fixture.n)
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        assert_ne!(expected, independently_repacked);
        let payload = source.component(&packed).unwrap();
        assert_eq!(payload.bytes(), expected);
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

    #[test]
    fn aggregate_experts_without_projection_axis_repack_independently() {
        let fixture = write_gate_up_fixture();
        let source = GptqMarlinSafetensorsSource::open(fixture.directory.path()).unwrap();
        let packed = WeightComponentSpec {
            id: WeightId::new("component.layer.experts.packed").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![
                "layer.gate.qweight".to_owned(),
                "layer.gate.qzeros".to_owned(),
                "layer.gate.g_idx".to_owned(),
                "layer.up.qweight".to_owned(),
                "layer.up.qzeros".to_owned(),
                "layer.up.g_idx".to_owned(),
            ],
            dimensions: vec![2, fixture.n as u64, (fixture.k / 2) as u64],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        };
        let expected = fixture
            .qweights
            .iter()
            .flat_map(|values| {
                repack_gptq_to_marlin(values, fixture.k, fixture.n)
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let payload = source.component(&packed).unwrap();
        assert_eq!(payload.bytes(), expected);
        assert_eq!(payload.dimensions(), packed.dimensions);

        let scales = WeightComponentSpec {
            id: WeightId::new("component.layer.experts.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec!["layer.gate.scales".to_owned(), "layer.up.scales".to_owned()],
            dimensions: vec![2, fixture.n as u64, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let expected_scales = fixture
            .scales
            .iter()
            .flat_map(|values| {
                repack_scales_to_marlin(values, 1, fixture.n, 1)
                    .into_iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
            })
            .collect::<Vec<_>>();
        let payload = source.component(&scales).unwrap();
        assert_eq!(payload.bytes(), expected_scales);
        assert_eq!(payload.dimensions(), scales.dimensions);
    }
}
