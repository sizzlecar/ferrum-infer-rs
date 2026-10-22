//! Usage: cargo run -p ferrum-models --features cuda --example generate_normal_noise -- CONFIG.json
//! CONFIG has seed, shape:{channels,height,width}, device, usable_memory_bytes,
//! bootstrap_memory_bytes, stream_memory_bytes, opaque_memory_profile,
//! maximum_host_bytes, output_directory. Every allowance is explicit.
//! The directory must not exist. Produces noise.nchw.bf16, noise.packed.bf16,
//! and receipt.json. These are reusable initial-noise artifacts, not evidence
//! that this CUDA toolkit matches deployment Torch's RNG bytes.

#[cfg(feature = "cuda")]
#[path = "generate_normal_noise/execution.rs"]
mod execution;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "support/tensor_component.rs"]
mod shared;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    {
        execution::main()
    }
    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("generate_normal_noise requires --features cuda")
    }
}
