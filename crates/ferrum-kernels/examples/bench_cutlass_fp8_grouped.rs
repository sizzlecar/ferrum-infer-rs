//! Same-input FP8 projection diagnostic. Explicit --run; no serving/SLO claim.
//! Uses one CUDA stream and a bounded set of diagnostic-owned allocations.
use std::{error::Error, fs::OpenOptions, io::Write};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[path = "bench_cutlass_fp8_grouped/arguments.rs"]
mod arguments;
#[cfg(feature = "cuda-cutlass-fp8")]
#[path = "bench_cutlass_fp8_grouped/execution.rs"]
mod execution;
#[cfg(any(feature = "cuda-cutlass-fp8", test))]
#[path = "bench_cutlass_fp8_grouped/fixture.rs"]
mod fixture;

fn main() -> Result<()> {
    let Some(args) = arguments::Arguments::parse(std::env::args().skip(1))? else {
        println!("{}", arguments::USAGE);
        return Ok(());
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    #[cfg(feature = "cuda-cutlass-fp8")]
    let result = execution::run(&args, &mut output);
    #[cfg(not(feature = "cuda-cutlass-fp8"))]
    let result: Result<()> =
        Err("requires cuda-cutlass-fp8 and its matching native artifact".into());
    if let Err(error) = &result {
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({"phase":"failed","error":error.to_string()}),
        )?;
        writeln!(output)?;
        output.flush()?;
    }
    result
}
