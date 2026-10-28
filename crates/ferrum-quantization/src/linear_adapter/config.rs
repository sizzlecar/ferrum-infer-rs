use super::{invalid, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

/// Scaling is a declared export convention, never inferred from the filename.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinearAdapterScaling {
    /// Export already has unit scaling, equivalent to alpha=rank.
    Unit,
    Alpha {
        alpha: f32,
    },
}
impl LinearAdapterScaling {
    pub(super) fn multiplier(self, rank: u64) -> Result<f32> {
        let value = match self {
            Self::Unit => 1.0,
            Self::Alpha { alpha } if alpha.is_finite() && alpha > 0.0 => alpha / rank as f32,
            _ => return Err(invalid("alpha must be finite and positive")),
        };
        if rank == 0 || !value.is_finite() || value <= 0.0 {
            return Err(invalid(
                "rank/scaling is outside the finite ordinary LoRA domain",
            ));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinearAdapterSource {
    Peft {
        directory: PathBuf,
        /// Required when PEFT's base_model_name_or_path is absent/null. An
        /// override cannot contradict an existing base-model declaration.
        #[serde(default)]
        base_model: Option<String>,
    },
    Diffusers {
        file: PathBuf,
        base_model: String,
        scaling: LinearAdapterScaling,
    },
}

pub(super) struct PeftConfig {
    pub base_model: Option<String>,
    pub rank: u64,
    pub scaling: LinearAdapterScaling,
    pub targets: Vec<String>,
}
impl PeftConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value: Value = serde_json::from_slice(bytes).map_err(invalid)?;
        let fields = value
            .as_object()
            .ok_or_else(|| invalid("PEFT config must be an object"))?;
        // Training-only and provenance fields are explicitly inert at inference.
        const ALLOWED: &[&str] = &[
            "peft_type",
            "r",
            "lora_alpha",
            "base_model_name_or_path",
            "target_modules",
            "bias",
            "fan_in_fan_out",
            "use_dora",
            "use_rslora",
            "use_qalora",
            "lora_bias",
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "loftq_config",
            "eva_config",
            "corda_config",
            "megatron_config",
            "target_parameters",
            "trainable_token_indices",
            "auto_mapping",
            "revision",
            "inference_mode",
            "task_type",
            "init_lora_weights",
            "lora_dropout",
            "layers_to_transform",
            "layers_pattern",
            "exclude_modules",
            "megatron_core",
            "qalora_group_size",
            "runtime_config",
            "layer_replication",
        ];
        for key in fields.keys() {
            if !ALLOWED.contains(&key.as_str()) {
                return Err(invalid(format!(
                    "unsupported PEFT configuration field {key:?}"
                )));
            }
        }
        if value["peft_type"] != "LORA" {
            return Err(invalid("only ordinary PEFT LORA is supported"));
        }
        for key in [
            "fan_in_fan_out",
            "use_dora",
            "use_rslora",
            "use_qalora",
            "lora_bias",
        ] {
            if fields.get(key).is_some_and(|v| !v.is_null() && *v != false) {
                return Err(invalid(format!("unsupported PEFT variant {key}")));
            }
        }
        if fields.get("bias").is_some_and(|v| *v != "none") {
            return Err(invalid("adapter bias is unsupported"));
        }
        for key in [
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "loftq_config",
            "eva_config",
            "corda_config",
            "megatron_config",
            "target_parameters",
            "trainable_token_indices",
            "runtime_config",
            "layer_replication",
        ] {
            if fields.get(key).is_some_and(|v| !empty(v)) {
                return Err(invalid(format!("unsupported PEFT extension {key}")));
            }
        }
        let rank = value["r"]
            .as_u64()
            .filter(|r| *r > 0)
            .ok_or_else(|| invalid("PEFT r must be a positive integer"))?;
        let alpha = value["lora_alpha"]
            .as_f64()
            .ok_or_else(|| invalid("PEFT lora_alpha is required"))? as f32;
        let scaling = LinearAdapterScaling::Alpha { alpha };
        scaling.multiplier(rank)?;
        let base_model = match fields.get("base_model_name_or_path") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) if !name.trim().is_empty() => Some(name.clone()),
            _ => return Err(invalid("invalid PEFT base_model_name_or_path")),
        };
        let targets = value["target_modules"]
            .as_array()
            .ok_or_else(|| invalid("PEFT target_modules requires an explicit string list"))?
            .iter()
            .map(|target| {
                target
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("invalid PEFT target module"))
            })
            .collect::<Result<Vec<_>>>()?;
        if targets.is_empty() {
            return Err(invalid("PEFT target_modules is empty"));
        }
        Ok(Self {
            base_model,
            rank,
            scaling,
            targets,
        })
    }
}
fn empty(value: &Value) -> bool {
    value.is_null()
        || value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(serde_json::Map::is_empty)
}
