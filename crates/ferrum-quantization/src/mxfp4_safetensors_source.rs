//! Exact source adapter for GPT-OSS safetensors MXFP4 expert weights.
//!
//! The checkpoint stores two E2M1 values per byte, low nibble first, in
//! groups of 32 values. Each group has one unsigned E8M0 exponent byte whose
//! multiplicative value is `2^(byte - 127)`. This adapter preserves those
//! bytes and their mmap ownership; execution layout conversion belongs to a
//! separately registered materializer.

use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, QuantizationPacking, QuantizationSpec, VNextError, WeightComponentPayload,
    WeightComponentRole, WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_types::Result;
use safetensors::Dtype;

use crate::safetensors_archive::{SafetensorsArchive, SafetensorsTensor};

pub const MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID: &str =
    "quantization.safetensors.mxfp4-e2m1-e8m0-group32-lsb-even";

/// Mmap-backed source for native GPT-OSS MXFP4 blocks and E8M0 scale bytes.
pub struct Mxfp4SafetensorsSource {
    archive: SafetensorsArchive,
}

impl Mxfp4SafetensorsSource {
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
        if component.dimensions.len() < 2 || component.dimensions.last() != Some(&16) {
            return Err(invalid_component(
                component,
                "MXFP4 blocks must end in 16 bytes representing 32 E2M1 values",
            ));
        }
        self.u8_payload(component, "_blocks", "MXFP4 blocks")
    }

    fn e8m0_scales<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.u8_payload(component, "_scales", "MXFP4 E8M0 scales")
    }

    fn u8_payload<'source>(
        &'source self,
        component: &WeightComponentSpec,
        required_suffix: &str,
        label: &str,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let tensor = self.single_tensor(component, required_suffix, label)?;
        let retained_host_memory = tensor.retained_host_memory().clone();
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            ElementType::U8,
            tensor.bytes(),
        )?
        .with_retained_host_memory(retained_host_memory)
    }

    fn single_tensor<'source>(
        &'source self,
        component: &WeightComponentSpec,
        required_suffix: &str,
        label: &str,
    ) -> std::result::Result<SafetensorsTensor<'source>, VNextError> {
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                format!("{label} require exactly one safetensors tensor"),
            ));
        };
        if !external_name.ends_with(required_suffix) {
            return Err(invalid_component(
                component,
                format!("{label} source must end with {required_suffix}"),
            ));
        }
        let tensor = self
            .archive
            .tensor(external_name)
            .map_err(|error| invalid_component(component, error.to_string()))?;
        if tensor.dtype() != Dtype::U8 {
            return Err(invalid_component(
                component,
                format!(
                    "{label} must use safetensors U8 storage, got {:?}",
                    tensor.dtype()
                ),
            ));
        }
        if tensor.shape() != component.dimensions {
            return Err(invalid_component(
                component,
                format!(
                    "{label} shape {:?} differs from typed shape {:?}",
                    tensor.shape(),
                    component.dimensions
                ),
            ));
        }
        Ok(tensor)
    }
}

impl WeightComponentSource for Mxfp4SafetensorsSource {
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
                    element_type: ElementType::U8,
                },
            ) => self.e8m0_scales(component),
            (WeightComponentRole::PackedValues | WeightComponentRole::Scales, _) => Err(
                invalid_component(component, "unsupported MXFP4 component encoding"),
            ),
            (_, WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. }) => {
                self.archive.component(component)
            }
            _ => Err(invalid_component(
                component,
                "unsupported MXFP4 component role or encoding",
            )),
        }
    }
}

fn validate_source_quantization(
    component: &WeightComponentSpec,
    quantization: &QuantizationSpec,
) -> std::result::Result<(), VNextError> {
    quantization.validate()?;
    if quantization.format_id.as_str() != MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.grouping.fixed_size() != Some(32)
        || quantization.packing != QuantizationPacking::Interleaved
        || quantization.scale_type != ElementType::U8
        || quantization.zero_point_type.is_some()
    {
        return Err(invalid_component(
            component,
            "typed MXFP4 source requires E2M1 low-nibble-first packing, group size 32, U8 E8M0 scales, and no zero points",
        ));
    }
    Ok(())
}

fn invalid_component(component: &WeightComponentSpec, reason: impl AsRef<str>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("MXFP4 component `{}`: {}", component.id, reason.as_ref()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ferrum_interfaces::vnext::{QuantizationFormatId, QuantizationGrouping, WeightId};
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    const BLOCKS_NAME: &str = "model.layers.0.mlp.experts.gate_up_proj_blocks";
    const SCALES_NAME: &str = "model.layers.0.mlp.experts.gate_up_proj_scales";
    const DENSE_NAME: &str = "model.layers.0.mlp.router.weight";

    fn quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID).unwrap(),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(32),
            packing: QuantizationPacking::Interleaved,
            scale_type: ElementType::U8,
            zero_point_type: None,
        }
    }

    fn blocks_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.mxfp4.blocks").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![BLOCKS_NAME.to_owned()],
            dimensions: vec![2, 4, 2, 16],
            encoding: WeightEncoding::Quantized(quantization()),
            required: true,
        }
    }

    fn scales_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.mxfp4.scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec![SCALES_NAME.to_owned()],
            dimensions: vec![2, 4, 2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
            required: true,
        }
    }

    fn dense_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.router.weight").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec![DENSE_NAME.to_owned()],
            dimensions: vec![4, 4],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        }
    }

    fn write_fixture(
        block_dtype: Dtype,
        scale_dtype: Dtype,
        scale_shape: [usize; 3],
    ) -> tempfile::TempDir {
        let directory = tempdir().unwrap();
        let blocks = vec![0x21_u8; 2 * 4 * 2 * 16];
        let scales = vec![127_u8; scale_shape.iter().product()];
        let dense = vec![0_u8; 4 * 4 * 2];
        let views = BTreeMap::from([
            (
                BLOCKS_NAME,
                TensorView::new(block_dtype, vec![2, 4, 2, 16], &blocks).unwrap(),
            ),
            (
                SCALES_NAME,
                TensorView::new(scale_dtype, scale_shape.to_vec(), &scales).unwrap(),
            ),
            (
                DENSE_NAME,
                TensorView::new(Dtype::BF16, vec![4, 4], &dense).unwrap(),
            ),
        ]);
        serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn component_error(
        source: &Mxfp4SafetensorsSource,
        component: &WeightComponentSpec,
    ) -> VNextError {
        match source.component(component) {
            Ok(_) => panic!("MXFP4 contract drift must fail closed"),
            Err(error) => error,
        }
    }

    #[test]
    fn exposes_native_mxfp4_blocks_and_e8m0_scales_without_copy() {
        let fixture = write_fixture(Dtype::U8, Dtype::U8, [2, 4, 2]);
        let source = Mxfp4SafetensorsSource::open(fixture.path()).unwrap();

        let blocks = source.component(&blocks_component()).unwrap();
        assert_eq!(blocks.element_type(), ElementType::U8);
        assert_eq!(blocks.dimensions(), [2, 4, 2, 16]);
        assert_eq!(blocks.bytes().len(), 256);
        assert!(blocks.retained_host_memory().is_some());
        assert_eq!(blocks.bytes()[0], 0x21);

        let scales = source.component(&scales_component()).unwrap();
        assert_eq!(scales.dimensions(), [2, 4, 2]);
        assert_eq!(scales.bytes(), &[127_u8; 16]);
        assert!(scales.retained_host_memory().is_some());

        let dense = source.component(&dense_component()).unwrap();
        assert_eq!(dense.element_type(), ElementType::Bf16);
        assert!(dense.retained_host_memory().is_some());
    }

    #[test]
    fn rejects_blocks_scale_shape_or_dtype_drift() {
        let bad_blocks = write_fixture(Dtype::I8, Dtype::U8, [2, 4, 2]);
        let source = Mxfp4SafetensorsSource::open(bad_blocks.path()).unwrap();
        let error = component_error(&source, &blocks_component());
        assert!(error.to_string().contains("safetensors U8"), "{error}");

        let bad_scales = write_fixture(Dtype::U8, Dtype::I8, [2, 4, 2]);
        let source = Mxfp4SafetensorsSource::open(bad_scales.path()).unwrap();
        let error = component_error(&source, &scales_component());
        assert!(error.to_string().contains("safetensors U8"), "{error}");

        let wrong_shape = write_fixture(Dtype::U8, Dtype::U8, [2, 4, 1]);
        let source = Mxfp4SafetensorsSource::open(wrong_shape.path()).unwrap();
        let error = component_error(&source, &scales_component());
        assert!(
            error.to_string().contains("differs from typed shape"),
            "{error}"
        );
    }

    #[test]
    fn rejects_noncanonical_mxfp4_contract() {
        let fixture = write_fixture(Dtype::U8, Dtype::U8, [2, 4, 2]);
        let source = Mxfp4SafetensorsSource::open(fixture.path()).unwrap();

        let mut wrong_group = blocks_component();
        let WeightEncoding::Quantized(spec) = &mut wrong_group.encoding else {
            unreachable!()
        };
        spec.grouping = QuantizationGrouping::fixed(64);
        let error = component_error(&source, &wrong_group);
        assert!(error.to_string().contains("group size 32"), "{error}");

        let mut wrong_suffix = scales_component();
        wrong_suffix.external_names[0] = "model.layers.0.mlp.experts.gate_up_proj.scale".to_owned();
        let error = component_error(&source, &wrong_suffix);
        assert!(
            error.to_string().contains("must end with _scales"),
            "{error}"
        );
    }
}
