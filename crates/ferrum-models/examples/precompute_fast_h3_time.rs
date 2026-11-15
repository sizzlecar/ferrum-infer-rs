//! Cold-path CUDA time MLP/AdaLN execution using the original checkpoint.
//! `--config PATH` accepts the typed JSON documented beside this example.
//! Outputs are neutral safetensors plus provenance, not a video or a speed claim.
#[path = "precompute_fast_h3_time/arguments.rs"]
mod arguments;
#[path = "precompute_fast_h3_time/artifacts.rs"]
mod artifacts;
#[cfg(feature = "cuda")]
#[path = "precompute_fast_h3_time/execution.rs"]
mod execution;
#[path = "support/fast_h3_package.rs"]
mod package;
#[path = "support/retained_failure.rs"]
mod retained_failure;

fn main() {
    use retained_failure::Failure;
    let result = (|| -> Result<(), Failure> {
        let mut cli = std::env::args().skip(1);
        if cli.next().as_deref() != Some("--config") {
            return Err(Failure::retain(
                "arguments",
                "usage: precompute_fast_h3_time --config PATH",
                (),
            ));
        }
        let path = cli
            .next()
            .ok_or_else(|| Failure::retain("arguments", "missing config path", ()))?;
        if cli.next().is_some() {
            return Err(Failure::retain(
                "arguments",
                "unexpected extra argument",
                (),
            ));
        }
        let args = arguments::Arguments::read(std::path::Path::new(&path))
            .map_err(|e| Failure::retain("configuration", e, ()))?;
        #[cfg(feature = "cuda")]
        {
            let report = execution::run(args)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| Failure::retain(
                    "report",
                    e,
                    ()
                ))?
            );
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = args;
            Err(Failure::retain("feature", "requires --features cuda", ()))
        }
    })();
    if let Err(failure) = result {
        std::mem::forget(failure.owner);
        eprintln!(
            "{}",
            serde_json::json!({"schema":"ferrum.fast_h3.time_precompute.failure.v1", "stage":failure.stage, "error":failure.message, "retirement":"not claimed; native failure and owner chain retained until process exit"})
        );
        std::process::exit(1);
    }
}
