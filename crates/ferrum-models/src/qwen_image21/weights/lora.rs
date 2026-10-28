//! Model-declared ordinary-linear adapter targets, including fused exports.
use std::ops::Range;

use ferrum_types::Result;

use super::{super::invalid, ExpectedTensor, ExpectedWeights};

/// One adapter matrix's logical [output, input] dimensions and complete routing.
/// This describes metadata only; it does not read or compose any weight payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraTarget<'weights> {
    dimensions: [u64; 2],
    slices: Vec<LoraTargetSlice<'weights>>,
}

/// Rows of the adapter's B matrix routed to one original checkpoint matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraTargetSlice<'weights> {
    tensor: &'weights ExpectedTensor,
    adapter_output_rows: Range<u64>,
}

impl LoraTarget<'_> {
    pub const fn dimensions(&self) -> [u64; 2] {
        self.dimensions
    }
    pub fn slices(&self) -> &[LoraTargetSlice<'_>] {
        &self.slices
    }
}

impl LoraTargetSlice<'_> {
    pub fn tensor(&self) -> &ExpectedTensor {
        self.tensor
    }
    pub fn adapter_output_rows(&self) -> Range<u64> {
        self.adapter_output_rows.clone()
    }
}

impl ExpectedWeights {
    /// Bind the complete imported adapter to this architecture before device
    /// creation. Turbo retains the same transformer target contract as 2.1;
    /// this structural check does not qualify an adapter's Turbo image quality.
    pub fn bind_lora(
        &self,
        imported: std::sync::Arc<ferrum_quantization::linear_adapter::ImportedLinearAdapter>,
        strength: f32,
    ) -> Result<ferrum_quantization::linear_adapter::PreparedLinearAdapter> {
        use ferrum_quantization::linear_adapter::{LinearAdapterBinding, PreparedLinearAdapter};
        if imported
            .architecture_hint()
            .is_some_and(|hint| hint != "qwen_image_2")
        {
            return Err(invalid(
                "adapter declares a different transformer architecture",
            ));
        }
        if !matches!(
            imported.base_model(),
            "Qwen/Qwen-Image-2.1" | "Qwen/Qwen-Image-2.1-Turbo"
        ) {
            return Err(invalid(
                "adapter base model is not a declared Qwen-Image-2.1 base",
            ));
        }
        let mut bindings = Vec::new();
        for (module, matrix) in imported.matrices() {
            let target = self.lora_target(module)?;
            if matrix.dimensions() != target.dimensions() {
                return Err(invalid(format!(
                    "adapter target {module:?} shape differs from model"
                )));
            }
            for slice in target.slices() {
                let dimensions = slice.tensor().dimensions();
                bindings.push(LinearAdapterBinding {
                    module: module.clone(),
                    external_weight: slice.tensor().external_name().to_owned(),
                    base_dimensions: [dimensions[0], dimensions[1]],
                    adapter_output_rows: slice.adapter_output_rows(),
                });
            }
        }
        PreparedLinearAdapter::new(imported, bindings, strength)
    }

    /// Resolve a canonical external module name, without a component prefix or
    /// `.weight` suffix. The format importer must normalize its declared naming
    /// convention first. Norms and unknown modules are not linear LoRA targets.
    pub fn lora_target(&self, module: &str) -> Result<LoraTarget<'_>> {
        if let Some(tensor) = self.tensors().get(&format!("{module}.weight")) {
            let [output, input] = tensor.dimensions() else {
                return Err(invalid(format!(
                    "adapter target {module:?} is not an ordinary linear matrix"
                )));
            };
            return Ok(LoraTarget {
                dimensions: [*output, *input],
                slices: vec![LoraTargetSlice {
                    tensor,
                    adapter_output_rows: 0..*output,
                }],
            });
        }
        if let Some(prefix) = module.strip_suffix(".img_mlp.gate_up") {
            let gate = self
                .tensors()
                .get(&format!("{prefix}.img_mlp.gate_layer.weight"));
            let up = self.tensors().get(&format!("{prefix}.img_mlp.proj.weight"));
            if let (Some(gate), Some(up)) = (gate, up) {
                let [rows, input] = gate.dimensions() else {
                    return Err(invalid("fused adapter gate is not a linear matrix"));
                };
                if up.dimensions() != gate.dimensions() {
                    return Err(invalid("fused adapter gate/up dimensions differ"));
                }
                let output = rows
                    .checked_mul(2)
                    .ok_or_else(|| invalid("fused adapter output rows overflow"))?;
                // Qwen-Image-2.1 fused SwiGLU exports store [gate; up]. Both
                // share A; split B's rows, preserving all of the adapter delta.
                return Ok(LoraTarget {
                    dimensions: [output, *input],
                    slices: vec![
                        LoraTargetSlice {
                            tensor: gate,
                            adapter_output_rows: 0..*rows,
                        },
                        LoraTargetSlice {
                            tensor: up,
                            adapter_output_rows: *rows..output,
                        },
                    ],
                });
            }
        }
        Err(invalid(format!("unknown linear adapter target {module:?}")))
    }
}
