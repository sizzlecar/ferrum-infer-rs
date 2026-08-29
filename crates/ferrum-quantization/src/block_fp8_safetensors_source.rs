//! Exact safetensors source adapter for rectangular block-FP8 checkpoints.
//!
//! This adapter preserves the checkpoint representation: E4M3 value bytes
//! and BF16 inverse scales are exposed as separate typed components without
//! decoding, requantizing, or selecting an execution kernel. A later,
//! explicitly registered materializer owns source-to-execution conversion.

use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, QuantizationPacking, QuantizationSpec, VNextError, WeightComponentPayload,
    WeightComponentRole, WeightComponentSource, WeightComponentSpec, WeightEncoding,
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
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "FP8 values require exactly one safetensors source",
            ));
        };
        if !external_name.ends_with(".weight") {
            return Err(invalid_component(
                component,
                "FP8 values source must end with .weight",
            ));
        }
        let tensor = self.tensor(component, external_name)?;
        if tensor.dtype() != Dtype::F8_E4M3 {
            return Err(invalid_component(
                component,
                format!(
                    "FP8 values must use safetensors F8_E4M3, got {:?}",
                    tensor.dtype()
                ),
            ));
        }
        validate_unit_prefix_shape(component, tensor.shape(), "FP8 values")?;
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            ElementType::U8,
            tensor.bytes(),
        )
    }

    fn inverse_scales<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "FP8 inverse scales require exactly one safetensors source",
            ));
        };
        if !external_name.ends_with(".weight_scale_inv") {
            return Err(invalid_component(
                component,
                "FP8 inverse-scale source must end with .weight_scale_inv",
            ));
        }
        let tensor = self.tensor(component, external_name)?;
        if tensor.dtype() != Dtype::BF16 {
            return Err(invalid_component(
                component,
                format!(
                    "FP8 inverse scales must use safetensors BF16, got {:?}",
                    tensor.dtype()
                ),
            ));
        }
        validate_unit_prefix_shape(component, tensor.shape(), "FP8 inverse scales")?;
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            ElementType::Bf16,
            tensor.bytes(),
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
