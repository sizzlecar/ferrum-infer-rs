//! Model config registry: ModelDefinition → TransformerConfig.
//!
//! Adding a new model = adding one function here (~10 lines).

mod llama;
mod mistral;
mod qwen3;
pub mod weight_loader;

use crate::definition::ModelDefinition;
use crate::registry::Architecture;
use ferrum_kernels::backend::TransformerConfig;

pub use llama::llama_config;
pub use mistral::mistral_config;
pub use qwen3::qwen3_config;

/// Convert a ModelDefinition into a TransformerConfig for the ModelRunner.
///
/// Returns `None` for non-decoder architectures (BERT, CLIP, Whisper, TTS).
pub fn to_transformer_config(def: &ModelDefinition) -> Option<TransformerConfig> {
    match def.architecture {
        Architecture::Qwen3 => Some(qwen3_config(def)),
        Architecture::Llama => Some(llama_config(def)),
        Architecture::Mistral => Some(mistral_config(def)),
        Architecture::Qwen2 => Some(llama_config(def)), // Qwen2 is structurally Llama-like
        // Non-decoder models: handled by their own executors
        Architecture::Bert
        | Architecture::Clip
        | Architecture::Whisper
        | Architecture::Phi
        | Architecture::GPT2
        | Architecture::Unknown => None,
    }
}
