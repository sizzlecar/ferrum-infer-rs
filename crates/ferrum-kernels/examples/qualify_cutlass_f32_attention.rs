//! Independent native qualification, not a model/backend capability declaration.
//! Build the locked native source separately; pass its library explicitly.
#[path = "qualify_cutlass_f32_attention/fixture.rs"]
mod fixture;
#[cfg(feature = "cuda")]
#[path = "qualify_cutlass_f32_attention/gpu.rs"]
mod gpu;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let mut library = None;
    let mut receipt = None;
    let mut iterations = 5;
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
        &library.ok_or("--library is required")?,
        &receipt.ok_or("--receipt is required")?,
        device,
        iterations,
    )
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    Err("qualification execution requires --features cuda; CPU fixture tests do not".into())
}
