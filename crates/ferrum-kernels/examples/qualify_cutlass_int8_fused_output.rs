//! Independent true-shape INT8 output qualification; never changes model routing.
#[cfg(any(feature = "cuda", test))]
#[path = "qualify_cutlass_int8_fused_output/fixture.rs"]
mod fixture;
#[cfg(feature = "cuda")]
#[path = "qualify_cutlass_int8_fused_output/gpu.rs"]
mod gpu;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let mut library = None;
    let mut receipt = None;
    let mut iterations = 3;
    let mut device = 0;
    let mut input = std::env::args().skip(1);
    while let Some(flag) = input.next() {
        let value = input.next().ok_or("missing argument value")?;
        match flag.as_str() {
            "--library" => library = Some(std::path::PathBuf::from(value)),
            "--receipt" => receipt = Some(std::path::PathBuf::from(value)),
            "--iterations" => iterations = value.parse()?,
            "--device" => device = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    gpu::run(
        &library.ok_or("--library required")?,
        &receipt.ok_or("--receipt required")?,
        device,
        iterations,
    )
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    Err("execution requires --features cuda; CPU fixture tests do not".into())
}
