//! Exact safetensors source adapter for rectangular block-FP8 checkpoints.
//!
//! This adapter preserves the checkpoint representation: E4M3 value bytes
//! and BF16 inverse scales are exposed as separate typed components without
//! decoding, requantizing, or selecting an execution kernel. A later,
//! explicitly registered materializer owns source-to-execution conversion.

use std::collections::BTreeSet;
use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, QuantizationPacking, QuantizationSpec, VNextError, WeightComponentPayload,
    WeightComponentRole, WeightComponentSegment, WeightComponentSegments, WeightComponentSource,
    WeightComponentSpec, WeightEncoding,
};
use ferrum_types::Result;
use safetensors::Dtype;

use crate::safetensors_archive::{SafetensorsArchive, SafetensorsTensor};

pub const BLOCK_FP8_E4M3_SOURCE_FORMAT_ID: &str =
    "quantization.safetensors.fp8-e4m3-block-grid-inverse-scale";

/// Mmap-backed block-FP8 checkpoint source. Dense exclusions retain the
/// archive's existing zero-copy/transcode behavior.
pub struct BlockFp8SafetensorsSource {
    archive: SafetensorsArchive,
}

impl BlockFp8SafetensorsSource {
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
        validate_source_quantization(component, quantization)?;
        self.ordered_matrix_payload(
            component,
            ".weight",
            Dtype::F8_E4M3,
            ElementType::U8,
            "FP8 values",
        )
    }

    fn packed_value_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
        quantization: &QuantizationSpec,
    ) -> std::result::Result<WeightComponentSegments<'source>, VNextError> {
        validate_source_quantization(component, quantization)?;
        self.ordered_matrix_segments(
            component,
            ".weight",
            Dtype::F8_E4M3,
            ElementType::U8,
            "FP8 values",
        )
    }

    fn inverse_scales<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.ordered_matrix_payload(
            component,
            ".weight_scale_inv",
            Dtype::BF16,
            ElementType::Bf16,
            "FP8 inverse scales",
        )
    }

    fn inverse_scale_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentSegments<'source>, VNextError> {
        self.ordered_matrix_segments(
            component,
            ".weight_scale_inv",
            Dtype::BF16,
            ElementType::Bf16,
            "FP8 inverse scales",
        )
    }

    fn ordered_matrix_payload<'source>(
        &'source self,
        component: &WeightComponentSpec,
        required_suffix: &str,
        required_dtype: Dtype,
        element_type: ElementType,
        label: &str,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let mut tensors =
            self.ordered_matrix_tensors(component, required_suffix, required_dtype, label)?;
        if tensors.len() == 1 {
            let tensor = tensors.pop().expect("one tensor was checked above");
            let retained_host_memory = tensor.retained_host_memory().clone();
            return WeightComponentPayload::new(
                component,
                tensor.external_name(),
                tensor.source_file(),
                component.dimensions.clone(),
                element_type,
                tensor.bytes(),
            )?
            .with_retained_host_memory(retained_host_memory);
        }

        let expected_bytes = usize::try_from(component.physical_bytes()?).map_err(|_| {
            invalid_component(
                component,
                format!("aggregate {label} byte size exceeds host address space"),
            )
        })?;
        let mut bytes = Vec::with_capacity(expected_bytes);
        let mut source_files = Vec::with_capacity(component.external_names.len());
        for tensor in tensors {
            source_files.push(tensor.source_file().to_owned());
            bytes.extend_from_slice(tensor.bytes());
        }
        WeightComponentPayload::from_ordered_sources(
            component,
            component.external_names.clone(),
            source_files,
            component.dimensions.clone(),
            element_type,
            bytes,
        )
    }

    fn ordered_matrix_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
        required_suffix: &str,
        required_dtype: Dtype,
        element_type: ElementType,
        label: &str,
    ) -> std::result::Result<WeightComponentSegments<'source>, VNextError> {
        let tensors =
            self.ordered_matrix_tensors(component, required_suffix, required_dtype, label)?;
        let mut source_files = Vec::with_capacity(tensors.len());
        let mut segments = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            source_files.push(tensor.source_file().to_owned());
            segments.push(
                WeightComponentSegment::new(tensor.bytes())
                    .with_retained_host_memory(tensor.retained_host_memory().clone())?,
            );
        }
        WeightComponentSegments::from_ordered_segments(
            component,
            component.external_names.clone(),
            source_files,
            component.dimensions.clone(),
            element_type,
            segments,
        )
    }

    fn ordered_matrix_tensors<'source>(
        &'source self,
        component: &WeightComponentSpec,
        required_suffix: &str,
        required_dtype: Dtype,
        label: &str,
    ) -> std::result::Result<Vec<SafetensorsTensor<'source>>, VNextError> {
        if component.external_names.is_empty() {
            return Err(invalid_component(
                component,
                format!("{label} require at least one safetensors source"),
            ));
        }
        let mut unique_names = BTreeSet::new();
        for external_name in &component.external_names {
            if !external_name.ends_with(required_suffix) {
                return Err(invalid_component(
                    component,
                    format!("every {label} source must end with {required_suffix}"),
                ));
            }
            if !unique_names.insert(external_name) {
                return Err(invalid_component(
                    component,
                    format!("ordered {label} sources contain duplicate tensor {external_name:?}"),
                ));
            }
        }

        if let [external_name] = component.external_names.as_slice() {
            let tensor = self.tensor(component, external_name)?;
            validate_source_tensor(component, &tensor, required_dtype, label)?;
            validate_unit_prefix_shape(component, tensor.shape(), label)?;
            return Ok(vec![tensor]);
        }

        let source_dimensions =
            aggregate_source_matrix_dimensions(component, component.external_names.len(), label)?;
        let mut tensors = Vec::with_capacity(component.external_names.len());
        for external_name in &component.external_names {
            let tensor = self.tensor(component, external_name)?;
            validate_source_tensor(component, &tensor, required_dtype, label)?;
            if tensor.shape() != source_dimensions {
                return Err(invalid_component(
                    component,
                    format!(
                        "ordered {label} source {external_name:?} shape {:?} differs from the typed matrix axes {source_dimensions:?}",
                        tensor.shape()
                    ),
                ));
            }
            tensors.push(tensor);
        }
        Ok(tensors)
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

fn validate_source_tensor(
    component: &WeightComponentSpec,
    tensor: &SafetensorsTensor<'_>,
    required_dtype: Dtype,
    label: &str,
) -> std::result::Result<(), VNextError> {
    if tensor.dtype() != required_dtype {
        return Err(invalid_component(
            component,
            format!(
                "{label} must use safetensors {required_dtype:?}, got {:?} for {:?}",
                tensor.dtype(),
                tensor.external_name()
            ),
        ));
    }
    Ok(())
}

fn aggregate_source_matrix_dimensions<'component>(
    component: &'component WeightComponentSpec,
    source_count: usize,
    label: &str,
) -> std::result::Result<&'component [u64], VNextError> {
    let matrix_axis = component.dimensions.len().checked_sub(2).ok_or_else(|| {
        invalid_component(
            component,
            format!("aggregate {label} require typed prefix plus matrix axes"),
        )
    })?;
    if matrix_axis == 0 {
        return Err(invalid_component(
            component,
            format!("aggregate {label} require at least one typed prefix axis"),
        ));
    }
    let expected_sources = component.dimensions[..matrix_axis]
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invalid_component(component, format!("aggregate {label} size overflows")))?;
    if usize::try_from(expected_sources).ok() != Some(source_count) {
        return Err(invalid_component(
            component,
            format!(
                "aggregate {label} typed prefix {:?} requires {expected_sources} ordered matrices, got {source_count}",
                &component.dimensions[..matrix_axis]
            ),
        ));
    }
    Ok(&component.dimensions[matrix_axis..])
}

impl WeightComponentSource for BlockFp8SafetensorsSource {
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
                    element_type: ElementType::Bf16,
                },
            ) => self.inverse_scales(component),
            (_, WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. }) => {
                self.archive.component(component)
            }
            _ => Err(invalid_component(
                component,
                "block-FP8 adapter received an unsupported component role or encoding",
            )),
        }
    }

    fn component_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentSegments<'source>, VNextError> {
        match (&component.role, &component.encoding) {
            (WeightComponentRole::PackedValues, WeightEncoding::Quantized(quantization)) => {
                self.packed_value_segments(component, quantization)
            }
            (
                WeightComponentRole::Scales,
                WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
            ) => self.inverse_scale_segments(component),
            (_, WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. }) => {
                self.archive.component_segments(component)
            }
            _ => Err(invalid_component(
                component,
                "block-FP8 adapter received an unsupported component role or encoding",
            )),
        }
    }
}

fn validate_source_quantization(
    component: &WeightComponentSpec,
    quantization: &QuantizationSpec,
) -> std::result::Result<(), VNextError> {
    quantization.validate()?;
    let block_shape = quantization
        .grouping
        .block_shape_2d()
        .map(|shape| [shape[0].get(), shape[1].get()]);
    if quantization.format_id.as_str() != BLOCK_FP8_E4M3_SOURCE_FORMAT_ID
        || quantization.bits_per_weight != 8
        || block_shape != Some([128, 128])
        || quantization.packing != QuantizationPacking::Linear
        || quantization.scale_type != ElementType::Bf16
        || quantization.zero_point_type.is_some()
    {
        return Err(invalid_component(
            component,
            "typed block-FP8 source requires linear E4M3 bytes, a 2D block grid, BF16 inverse scales, and no zero points",
        ));
    }
    Ok(())
}

fn validate_unit_prefix_shape(
    component: &WeightComponentSpec,
    source_shape: &[u64],
    label: &str,
) -> std::result::Result<(), VNextError> {
    let prefix_len = component
        .dimensions
        .len()
        .checked_sub(source_shape.len())
        .ok_or_else(|| invalid_component(component, format!("{label} source rank is too large")))?;
    if component.dimensions[prefix_len..] != *source_shape
        || component.dimensions[..prefix_len]
            .iter()
            .any(|extent| *extent != 1)
    {
        return Err(invalid_component(
            component,
            format!(
                "{label} source shape {source_shape:?} differs from typed shape {:?}",
                component.dimensions
            ),
        ));
    }
    Ok(())
}

fn invalid_component(component: &WeightComponentSpec, reason: impl AsRef<str>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!(
            "block-FP8 component `{}`: {}",
            component.id,
            reason.as_ref()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    use ferrum_interfaces::vnext::{QuantizationFormatId, QuantizationGrouping, WeightId};
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    const VALUES_NAME: &str = "model.layers.0.mlp.gate_proj.weight";
    const SCALES_NAME: &str = "model.layers.0.mlp.gate_proj.weight_scale_inv";
    const ORDERED_VALUE_NAMES: [&str; 4] = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    ];
    const ORDERED_SCALE_NAMES: [&str; 4] = [
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.up_proj.weight_scale_inv",
    ];

    fn quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(BLOCK_FP8_E4M3_SOURCE_FORMAT_ID).unwrap(),
            bits_per_weight: 8,
            grouping: QuantizationGrouping::block_2d([
                NonZeroU32::new(128).unwrap(),
                NonZeroU32::new(128).unwrap(),
            ]),
            packing: QuantizationPacking::Linear,
            scale_type: ElementType::Bf16,
            zero_point_type: None,
        }
    }

    fn write_fixture(value_dtype: Dtype, scale_dtype: Dtype) -> tempfile::TempDir {
        let directory = tempdir().unwrap();
        let n = 130_usize;
        let k = 257_usize;
        let value_element_bytes = match value_dtype {
            Dtype::F8_E4M3 | Dtype::U8 => 1,
            Dtype::F16 | Dtype::BF16 => 2,
            other => panic!("unsupported test value dtype {other:?}"),
        };
        let scale_element_bytes = match scale_dtype {
            Dtype::F16 | Dtype::BF16 => 2,
            other => panic!("unsupported test scale dtype {other:?}"),
        };
        let values = vec![0x38_u8; n * k * value_element_bytes];
        let scales = vec![0_u8; 2 * 3 * scale_element_bytes];
        let views = BTreeMap::from([
            (
                VALUES_NAME,
                TensorView::new(value_dtype, vec![n, k], &values).unwrap(),
            ),
            (
                SCALES_NAME,
                TensorView::new(scale_dtype, vec![2, 3], &scales).unwrap(),
            ),
        ]);
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn write_ordered_fixture(last_scale_shape: [usize; 2]) -> tempfile::TempDir {
        let directory = tempdir().unwrap();
        let value_0 = [1_u8; 6];
        let value_1 = [2_u8; 6];
        let value_2 = [3_u8; 6];
        let value_3 = [4_u8; 6];
        let scale_0 = [10_u8, 0];
        let scale_1 = [20_u8, 0];
        let scale_2 = [30_u8, 0];
        let scale_3 = vec![40_u8; last_scale_shape[0] * last_scale_shape[1] * 2];
        let views = BTreeMap::from([
            (
                ORDERED_VALUE_NAMES[0],
                TensorView::new(Dtype::F8_E4M3, vec![2, 3], &value_0).unwrap(),
            ),
            (
                ORDERED_VALUE_NAMES[1],
                TensorView::new(Dtype::F8_E4M3, vec![2, 3], &value_1).unwrap(),
            ),
            (
                ORDERED_VALUE_NAMES[2],
                TensorView::new(Dtype::F8_E4M3, vec![2, 3], &value_2).unwrap(),
            ),
            (
                ORDERED_VALUE_NAMES[3],
                TensorView::new(Dtype::F8_E4M3, vec![2, 3], &value_3).unwrap(),
            ),
            (
                ORDERED_SCALE_NAMES[0],
                TensorView::new(Dtype::BF16, vec![1, 1], &scale_0).unwrap(),
            ),
            (
                ORDERED_SCALE_NAMES[1],
                TensorView::new(Dtype::BF16, vec![1, 1], &scale_1).unwrap(),
            ),
            (
                ORDERED_SCALE_NAMES[2],
                TensorView::new(Dtype::BF16, vec![1, 1], &scale_2).unwrap(),
            ),
            (
                ORDERED_SCALE_NAMES[3],
                TensorView::new(Dtype::BF16, last_scale_shape.to_vec(), &scale_3).unwrap(),
            ),
        ]);
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn values_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.fp8.values").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![VALUES_NAME.to_owned()],
            dimensions: vec![1, 130, 257],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        }
    }

    fn scales_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.fp8.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec![SCALES_NAME.to_owned()],
            dimensions: vec![1, 2, 3],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        }
    }

    fn ordered_values_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.fp8.expert_gate_up.values").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: ORDERED_VALUE_NAMES.map(str::to_owned).to_vec(),
            dimensions: vec![2, 2, 2, 3],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        }
    }

    fn ordered_scales_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.fp8.expert_gate_up.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: ORDERED_SCALE_NAMES.map(str::to_owned).to_vec(),
            dimensions: vec![2, 2, 1, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        }
    }

    #[test]
    fn exposes_exact_e4m3_values_and_bf16_inverse_scales_with_unit_prefix_reshape() {
        let fixture = write_fixture(Dtype::F8_E4M3, Dtype::BF16);
        let source = BlockFp8SafetensorsSource::open(fixture.path()).unwrap();

        let values = source.component(&values_component()).unwrap();
        let scales = source.component(&scales_component()).unwrap();

        assert_eq!(values.dimensions(), [1, 130, 257]);
        assert_eq!(values.element_type(), ElementType::U8);
        assert_eq!(values.bytes().len(), 130 * 257);
        assert_eq!(scales.dimensions(), [1, 2, 3]);
        assert_eq!(scales.element_type(), ElementType::Bf16);
        assert_eq!(scales.bytes().len(), 2 * 3 * 2);
        assert!(values.retained_host_memory().is_some());
        assert!(scales.retained_host_memory().is_some());
        assert!(std::ptr::eq(
            values.bytes().as_ptr(),
            source
                .archive()
                .tensor(VALUES_NAME)
                .unwrap()
                .bytes()
                .as_ptr()
        ));
    }

    #[test]
    fn aggregates_expert_major_matrices_in_exact_schema_order() {
        let fixture = write_ordered_fixture([1, 1]);
        let source = BlockFp8SafetensorsSource::open(fixture.path()).unwrap();

        let values = source.component(&ordered_values_component()).unwrap();
        let scales = source.component(&ordered_scales_component()).unwrap();

        assert_eq!(
            values.external_names(),
            ORDERED_VALUE_NAMES.map(str::to_owned)
        );
        assert_eq!(values.dimensions(), [2, 2, 2, 3]);
        assert_eq!(
            values.bytes(),
            [vec![1_u8; 6], vec![2_u8; 6], vec![3_u8; 6], vec![4_u8; 6]].concat()
        );
        assert_eq!(
            scales.external_names(),
            ORDERED_SCALE_NAMES.map(str::to_owned)
        );
        assert_eq!(scales.dimensions(), [2, 2, 1, 1]);
        assert_eq!(scales.bytes(), [10, 0, 20, 0, 30, 0, 40, 40]);
    }

    #[test]
    fn exposes_ordered_retained_mmap_segments_without_aggregate_copy() {
        let fixture = write_ordered_fixture([1, 1]);
        let source = BlockFp8SafetensorsSource::open(fixture.path()).unwrap();

        let values = source
            .component_segments(&ordered_values_component())
            .unwrap();
        let scales = source
            .component_segments(&ordered_scales_component())
            .unwrap();

        assert_eq!(
            values.external_names(),
            ORDERED_VALUE_NAMES.map(str::to_owned)
        );
        assert_eq!(values.source_files(), ["model.safetensors"; 4]);
        assert_eq!(values.dimensions(), [2, 2, 2, 3]);
        assert_eq!(values.element_type(), ElementType::U8);
        assert_eq!(values.total_bytes(), 24);
        assert_eq!(values.segments().len(), 4);
        assert_eq!(
            scales.external_names(),
            ORDERED_SCALE_NAMES.map(str::to_owned)
        );
        assert_eq!(scales.source_files(), ["model.safetensors"; 4]);
        assert_eq!(scales.dimensions(), [2, 2, 1, 1]);
        assert_eq!(scales.element_type(), ElementType::Bf16);
        assert_eq!(scales.total_bytes(), 8);
        assert_eq!(scales.segments().len(), 4);

        for (index, external_name) in ORDERED_VALUE_NAMES.iter().enumerate() {
            let tensor = source.archive().tensor(external_name).unwrap();
            let segment = &values.segments()[index];
            assert!(std::ptr::eq(
                segment.bytes().as_ptr(),
                tensor.bytes().as_ptr()
            ));
            assert!(segment.retained_host_memory().is_some());
            assert_eq!(segment.bytes(), vec![u8::try_from(index + 1).unwrap(); 6]);
        }
        for (index, external_name) in ORDERED_SCALE_NAMES.iter().enumerate() {
            let tensor = source.archive().tensor(external_name).unwrap();
            let segment = &scales.segments()[index];
            assert!(std::ptr::eq(
                segment.bytes().as_ptr(),
                tensor.bytes().as_ptr()
            ));
            assert!(segment.retained_host_memory().is_some());
        }
        assert_eq!(scales.segments()[0].bytes(), [10, 0]);
        assert_eq!(scales.segments()[1].bytes(), [20, 0]);
        assert_eq!(scales.segments()[2].bytes(), [30, 0]);
        assert_eq!(scales.segments()[3].bytes(), [40, 40]);
    }

    #[test]
    fn rejects_aggregate_prefix_sidecar_or_grid_drift() {
        let fixture = write_ordered_fixture([1, 1]);
        let source = BlockFp8SafetensorsSource::open(fixture.path()).unwrap();

        let mut wrong_prefix = ordered_values_component();
        wrong_prefix.dimensions[1] = 3;
        let error = match source.component(&wrong_prefix) {
            Ok(_) => panic!("aggregate prefix drift must fail closed"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("requires 6 ordered matrices"),
            "{error}"
        );

        let mut wrong_sidecar = ordered_scales_component();
        wrong_sidecar.external_names[3] = "model.layers.0.mlp.experts.1.up_proj.scale".to_owned();
        let error = match source.component(&wrong_sidecar) {
            Ok(_) => panic!("aggregate sidecar drift must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("weight_scale_inv"), "{error}");

        let bad_grid = write_ordered_fixture([1, 2]);
        let source = BlockFp8SafetensorsSource::open(bad_grid.path()).unwrap();
        let error = match source.component(&ordered_scales_component()) {
            Ok(_) => panic!("aggregate scale-grid drift must fail closed"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("typed matrix axes [1, 1]"),
            "{error}"
        );
    }

    #[test]
    fn rejects_value_or_scale_dtype_drift() {
        let bad_values = write_fixture(Dtype::U8, Dtype::BF16);
        let source = BlockFp8SafetensorsSource::open(bad_values.path()).unwrap();
        let error = match source.component(&values_component()) {
            Ok(_) => panic!("non-E4M3 value dtype must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("F8_E4M3"), "{error}");

        let bad_scales = write_fixture(Dtype::F8_E4M3, Dtype::F16);
        let source = BlockFp8SafetensorsSource::open(bad_scales.path()).unwrap();
        let error = match source.component(&scales_component()) {
            Ok(_) => panic!("non-BF16 inverse-scale dtype must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("BF16"), "{error}");
    }

    #[test]
    fn rejects_noncanonical_source_quantization_contract() {
        let fixture = write_fixture(Dtype::F8_E4M3, Dtype::BF16);
        let source = BlockFp8SafetensorsSource::open(fixture.path()).unwrap();
        let mut component = values_component();
        let WeightEncoding::Quantized(spec) = &mut component.encoding else {
            unreachable!()
        };
        spec.packing = QuantizationPacking::Tiled;

        let error = match source.component(&component) {
            Ok(_) => panic!("noncanonical source quantization must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("linear E4M3"), "{error}");

        let mut component = values_component();
        let WeightEncoding::Quantized(spec) = &mut component.encoding else {
            unreachable!()
        };
        spec.grouping = QuantizationGrouping::block_2d([
            NonZeroU32::new(64).unwrap(),
            NonZeroU32::new(128).unwrap(),
        ]);

        let error = match source.component(&component) {
            Ok(_) => panic!("noncanonical block shape must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("linear E4M3"), "{error}");
    }
}
