//! `Linear<B>` trait — weight-bearing projection abstraction.
//!
//! Lives in ferrum-kernels alongside `Backend` because:
//!   1. `Backend::layer_forward_fused` and other "standard transformer layer"
//!      helpers want to accept `&dyn Linear<Self>` as their projection
//!      parameter, so the trait must be visible here.
//!   2. Model code in `ferrum-models` depends on both ferrum-kernels and
//!      ferrum-quantization, so keeping the trait in kernels avoids any
//!      circular dependency between kernels and quantization.
//!
//! Concrete implementations (DenseLinear, GptqLinear, AwqLinear, GgufLinear)
//! live in `ferrum-quantization`, which depends on `ferrum-kernels` for this
//! trait and for the `Backend` it parameterises over.

use crate::backend::Backend;

/// Stable projection role metadata derived from model weight names.
///
/// This is intentionally small and backend-neutral. It lets product code
/// choose a typed optimization path without depending on profiling labels or
/// hidden environment variables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearProjectionRole {
    Qkv,
    Query,
    Key,
    Value,
    Output,
    GateUp,
    Gate,
    Up,
    Down,
    LmHead,
}

/// Optional metadata carried by a loaded linear projection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinearMetadata {
    pub layer_index: Option<usize>,
    pub role: Option<LinearProjectionRole>,
}

impl LinearMetadata {
    pub const fn new(layer_index: Option<usize>, role: Option<LinearProjectionRole>) -> Self {
        Self { layer_index, role }
    }

    pub const fn is_empty(self) -> bool {
        self.layer_index.is_none() && self.role.is_none()
    }

    pub fn from_name(name: &str) -> Self {
        let base = strip_tensor_suffix(name);
        Self {
            layer_index: parse_layer_index(base),
            role: parse_projection_role(base),
        }
    }

    pub fn from_fused_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut layer_index = None;
        let mut roles = Vec::new();

        for name in names {
            let metadata = Self::from_name(name);
            if layer_index.is_none() {
                layer_index = metadata.layer_index;
            }
            if let Some(role) = metadata.role {
                roles.push(role);
            }
        }

        let role = match roles.as_slice() {
            [LinearProjectionRole::Query, LinearProjectionRole::Key, LinearProjectionRole::Value] => {
                Some(LinearProjectionRole::Qkv)
            }
            [LinearProjectionRole::Gate, LinearProjectionRole::Up] => {
                Some(LinearProjectionRole::GateUp)
            }
            [single] => Some(*single),
            _ => None,
        };

        Self { layer_index, role }
    }
}

/// A weight-bearing linear projection.
///
/// `forward` computes `out[m, out_features] = input[m, in_features] @ W^T`.
/// Implementations are responsible for calling the right backend kernel
/// (`B::gemm` for dense, `B::gemm_quant` for quantized variants).
pub trait Linear<B: Backend>: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;

    fn metadata(&self) -> LinearMetadata {
        LinearMetadata::default()
    }

    /// Append GEMM work onto `ctx`. Caller flushes the context when results
    /// must be materialised.
    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize);
}

fn strip_tensor_suffix(name: &str) -> &str {
    for suffix in [
        ".weight", ".qweight", ".scales", ".qzeros", ".g_idx", ".bias",
    ] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped;
        }
    }
    name
}

fn parse_layer_index(name: &str) -> Option<usize> {
    let mut prev_was_layers = false;
    for part in name.split('.') {
        if prev_was_layers {
            return part.parse::<usize>().ok();
        }
        prev_was_layers = part == "layers";
    }
    None
}

fn parse_projection_role(name: &str) -> Option<LinearProjectionRole> {
    let tail = name.rsplit('.').next().unwrap_or(name);
    match tail {
        "qkv_proj" => Some(LinearProjectionRole::Qkv),
        "q_proj" => Some(LinearProjectionRole::Query),
        "k_proj" => Some(LinearProjectionRole::Key),
        "v_proj" => Some(LinearProjectionRole::Value),
        "o_proj" => Some(LinearProjectionRole::Output),
        "gate_up_proj" => Some(LinearProjectionRole::GateUp),
        "gate_proj" => Some(LinearProjectionRole::Gate),
        "up_proj" => Some(LinearProjectionRole::Up),
        "down_proj" => Some(LinearProjectionRole::Down),
        "lm_head" => Some(LinearProjectionRole::LmHead),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{LinearMetadata, LinearProjectionRole};

    #[test]
    fn metadata_parses_llama_layer_projection_roles() {
        assert_eq!(
            LinearMetadata::from_name("model.layers.17.mlp.down_proj.qweight"),
            LinearMetadata::new(Some(17), Some(LinearProjectionRole::Down))
        );
        assert_eq!(
            LinearMetadata::from_name("language_model.model.layers.3.self_attn.o_proj.weight"),
            LinearMetadata::new(Some(3), Some(LinearProjectionRole::Output))
        );
        assert_eq!(
            LinearMetadata::from_name("lm_head.weight"),
            LinearMetadata::new(None, Some(LinearProjectionRole::LmHead))
        );
    }

    #[test]
    fn metadata_parses_fused_projection_roles() {
        assert_eq!(
            LinearMetadata::from_fused_names([
                "model.layers.2.self_attn.q_proj",
                "model.layers.2.self_attn.k_proj",
                "model.layers.2.self_attn.v_proj",
            ]),
            LinearMetadata::new(Some(2), Some(LinearProjectionRole::Qkv))
        );
        assert_eq!(
            LinearMetadata::from_fused_names([
                "model.layers.9.mlp.gate_proj.qweight",
                "model.layers.9.mlp.up_proj.qweight",
            ]),
            LinearMetadata::new(Some(9), Some(LinearProjectionRole::GateUp))
        );
    }
}
