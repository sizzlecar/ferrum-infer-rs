//! Bind an unfused checkpoint inventory to the architecture's logical weights.
use super::*;
use ferrum_quantization::TensorArchive;
use ferrum_runtime::{PhysicalWeightComponentBinding, PhysicalWeightPadding};

impl ExpectedWeights {
    /// A complete native inventory, or an empty map for the original exact
    /// BF16 layout. Shape and encoding are checked before any backend exists.
    /// Dense F32/F16 entries declare cold normalization to the logical BF16
    /// type; callers must provide a managed source for that conversion.
    pub fn archive_encodings(
        &self,
        archive: &dyn TensorArchive,
    ) -> Result<BTreeMap<String, WeightEncoding>> {
        let mut encodings = BTreeMap::new();
        for expected in self.tensors.values() {
            let tensor = archive.tensor(expected.external_name())?;
            if tensor.shape() != expected.dimensions() {
                return Err(invalid(format!(
                    "native archive shape differs: {}",
                    expected.external_name()
                )));
            }
            encodings.insert(
                expected.external_name().to_owned(),
                tensor.encoding().clone(),
            );
        }
        if encodings.values().all(|encoding| {
            *encoding
                == (WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                })
        }) {
            return Ok(BTreeMap::new());
        }
        // Fused or prefixed checkpoint layouts cannot be partially matched.
        if archive.tensor_count() != self.tensors.len()
            || archive
                .tensor_names()
                .any(|name| !self.tensors.contains_key(name))
        {
            return Err(invalid(
                "native archive must contain exactly the declared unfused tensor inventory",
            ));
        }
        self.weight_schema_with_encodings(&encodings)?;
        Ok(encodings)
    }

    /// Preserve native blocks in the physical schema. Logical dimensions and
    /// BF16 activation boundaries stay independent from storage precision.
    pub fn weight_schema_with_encodings(
        &self,
        encodings: &BTreeMap<String, WeightEncoding>,
    ) -> Result<WeightSchema> {
        let mut schema = self.weight_schema()?;
        if encodings.is_empty() {
            return Ok(schema);
        }
        if encodings.len() != self.tensors.len()
            || encodings
                .keys()
                .any(|name| !self.tensors.contains_key(name))
        {
            return Err(invalid(
                "native encoding map must bind every declared tensor exactly once",
            ));
        }
        schema.format_id = WeightFormatId::new("weight-format.gguf.native-block")
            .map_err(|e| invalid(e.to_string()))?;
        schema.layout_id =
            WeightLayoutId::new("weight-layout.qwen_image21.transformer.native-block.bf16.v1")
                .map_err(|e| invalid(e.to_string()))?;
        for (component, tensor) in schema.components.iter_mut().zip(&mut schema.tensors) {
            let name = &component.external_names[0];
            match &encodings[name] {
                WeightEncoding::Dense {
                    element_type: ElementType::Bf16 | ElementType::F16 | ElementType::F32,
                } => {}
                WeightEncoding::BlockQuantized(block) => {
                    block.validate().map_err(|e| invalid(e.to_string()))?;
                    let [_, width] = tensor.dimensions.as_slice() else {
                        return Err(invalid(format!(
                            "only matrix projections accept native blocks: {name}"
                        )));
                    };
                    if !width.is_multiple_of(u64::from(block.logical_values_per_block)) {
                        return Err(invalid(format!(
                            "native matrix rows must contain complete blocks: {name}"
                        )));
                    }
                    component.role = WeightComponentRole::PackedValues;
                    component.dimensions[1] /= u64::from(block.logical_values_per_block);
                    component.encoding = encodings[name].clone();
                    tensor.physical_layout = PhysicalWeightLayout::BlockQuantized {
                        blocks: PhysicalWeightComponentBinding::exact_contiguous(
                            component.id.clone(),
                        ),
                        block_axis: 1,
                        block_padding: PhysicalWeightPadding::Exact,
                    };
                }
                _ => {
                    return Err(invalid(format!(
                        "unsupported native source encoding: {name}"
                    )))
                }
            }
        }
        schema
            .validate(&ModelFamilyId::new(FAMILY_ID).map_err(|e| invalid(e.to_string()))?)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(schema)
    }
}
