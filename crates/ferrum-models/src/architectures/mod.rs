//! Model architecture implementations

pub mod bert;
pub mod llama;
pub mod mistral;
pub mod qwen2;
pub mod qwen3;

pub use bert::BertModelWrapper;
pub use llama::LlamaModelWrapper;
pub use mistral::MistralModelWrapper;
pub use qwen2::Qwen2ModelWrapper;
pub use qwen3::Qwen3ModelWrapper;

/// GQA repeat_kv: repeat K/V heads to match Q heads.
pub(crate) fn repeat_kv(
    x: candle_core::Tensor,
    n_rep: usize,
) -> candle_core::Result<candle_core::Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, nkv, seq, hd) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, nkv, n_rep, seq, hd))?;
    x.reshape((b, nkv * n_rep, seq, hd))
}
