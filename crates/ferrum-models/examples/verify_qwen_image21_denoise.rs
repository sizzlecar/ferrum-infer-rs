//! Repeated, managed DiT -> BF16 Euler over explicit checkpoint sigmas.
//! Caller supplies the initial latent; this example neither creates RNG state
//! nor claims a decoded image or full-trajectory deployment parity.
#[path = "verify_qwen_image21_denoise/arguments.rs"]
mod arguments;
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_denoise/composition.rs"]
mod composition;
#[path = "verify_qwen_image21_denoise/entry.rs"]
mod entry;
#[path = "verify_qwen_image21_denoise/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_denoise/inputs.rs"]
mod inputs;
#[path = "verify_qwen_image21_denoise/schedule.rs"]
mod schedule;
#[path = "support/tensor_component.rs"]
mod shared;
fn main() -> anyhow::Result<()> {
    entry::main()
}
#[cfg(test)]
#[path = "verify_qwen_image21_denoise/tests.rs"]
mod tests;
