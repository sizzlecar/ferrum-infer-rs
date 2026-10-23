use std::collections::BTreeMap;

use ferrum_runtime::{
    ContractVersion, ElementType, ModelFamilyId, PhysicalWeightLayout, WeightComponentRole,
    WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightSchema,
    WeightTensorSpec,
};
use ferrum_types::Result;
use serde::Serialize;

use super::{invalid, QwenImage21Config, StructureLimits, FAMILY_ID};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpectedTensor {
    external_name: String,
    dimensions: Vec<u64>,
    element_type: ElementType,
    bytes: u64,
}

impl ExpectedTensor {
    pub fn external_name(&self) -> &str {
        &self.external_name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Expected physical BF16 structure, derived from checked semantic dimensions.
/// This is neither a payload source nor proof that any checkpoint was read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpectedWeights {
    tensors: BTreeMap<String, ExpectedTensor>,
    payload_bytes: u64,
}

impl ExpectedWeights {
    pub fn from_config(config: &QwenImage21Config, limits: StructureLimits) -> Result<Self> {
        Self::with_tensor_limit(config, limits.maximum_tensors())
    }

    pub(super) fn with_tensor_limit(
        config: &QwenImage21Config,
        maximum_tensors: usize,
    ) -> Result<Self> {
        let count = config.layers() * 9 + 9;
        if count > maximum_tensors as u64 {
            return Err(invalid(
                "expected transformer tensor count exceeds inventory limit",
            ));
        }
        let mut tensors = BTreeMap::new();
        let h = config.hidden_size();
        let m = config.mlp_hidden_size();
        let mut append = |name: String, dimensions: Vec<u64>| -> Result<()> {
            let bytes = dimensions
                .iter()
                .try_fold(2_u64, |n, d| n.checked_mul(*d))
                .ok_or_else(|| invalid("tensor byte size overflows"))?;
            let tensor = ExpectedTensor {
                external_name: name.clone(),
                dimensions,
                element_type: ElementType::Bf16,
                bytes,
            };
            if tensors.insert(name, tensor).is_some() {
                return Err(invalid("duplicate expected tensor"));
            }
            Ok(())
        };
        for (name, shape) in [
            ("img_in.weight", vec![h, config.in_channels()]),
            ("modulation.1.weight", vec![4 * h, h]),
            (
                "time_text_embed.timestep_embedder.linear_1.weight",
                vec![h, 256],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.weight",
                vec![h, h],
            ),
            ("txt_in.text_norm.weight", vec![config.context_dim()]),
            ("txt_in.in_layer.weight", vec![h, config.context_dim()]),
            ("txt_in.out_layer.weight", vec![h, h]),
            ("norm_out.linear.weight", vec![h, h]),
            ("proj_out.weight", vec![config.out_channels(), h]),
        ] {
            append(name.to_owned(), shape)?;
        }
        for layer in 0..config.layers() {
            for (suffix, shape) in [
                ("attn.to_q.weight", vec![h, h]),
                ("attn.to_k.weight", vec![h, h]),
                ("attn.to_v.weight", vec![h, h]),
                ("attn.to_out.0.weight", vec![h, h]),
                ("attn.norm_q.weight", vec![config.head_dim()]),
                ("attn.norm_k.weight", vec![config.head_dim()]),
                ("img_mlp.gate_layer.weight", vec![m, h]),
                ("img_mlp.proj.weight", vec![m, h]),
                ("img_mlp.out.weight", vec![h, m]),
            ] {
                append(format!("transformer_blocks.{layer}.{suffix}"), shape)?;
            }
        }
        let payload_bytes = tensors
            .values()
            .try_fold(0_u64, |n, t| n.checked_add(t.bytes))
            .ok_or_else(|| invalid("transformer total bytes overflow"))?;
        Ok(Self {
            tensors,
            payload_bytes,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, ExpectedTensor> {
        &self.tensors
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub const fn parameter_count(&self) -> u64 {
        self.payload_bytes / 2
    }

    /// Runtime physical schema, with each original checkpoint tensor retained
    /// as its own component. Packing and numerical specialization come later.
    pub fn weight_schema(&self) -> Result<WeightSchema> {
        let mut components = Vec::with_capacity(self.tensors.len());
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for expected in self.tensors.values() {
            let component_id = WeightId::new(format!(
                "weight.component.qwen_image21.{}",
                expected.external_name
            ))
            .map_err(|e| invalid(e.to_string()))?;
            components.push(WeightComponentSpec {
                id: component_id.clone(),
                role: WeightComponentRole::Values,
                external_names: vec![expected.external_name.clone()],
                dimensions: expected.dimensions.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
                required: true,
            });
            tensors.push(WeightTensorSpec {
                id: WeightId::new(format!("weight.qwen_image21.{}", expected.external_name))
                    .map_err(|e| invalid(e.to_string()))?,
                dimensions: expected.dimensions.clone(),
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense { component_id },
                required: true,
            });
        }
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")
                .map_err(|e| invalid(e.to_string()))?,
            layout_id: WeightLayoutId::new("weight-layout.qwen_image21.transformer.bf16")
                .map_err(|e| invalid(e.to_string()))?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        };
        schema
            .validate(&ModelFamilyId::new(FAMILY_ID).map_err(|e| invalid(e.to_string()))?)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(schema)
    }
}
