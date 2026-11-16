//! Full-checkpoint DiT and PDD consumer. Both workloads reuse one static
//! backbone plan; PDD advances the shared typed continuation from explicit inputs.
#[path = "execute_fast_h3_dit/config.rs"]
mod config;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
#[path = "execute_fast_h3_dit/execution.rs"]
mod execution;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
#[path = "execute_fast_h3_dit/limits.rs"]
mod limits;
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
                "usage: execute_fast_h3_dit --config PATH",
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
        let config = config::Config::read(std::path::Path::new(&path))
            .map_err(|e| Failure::retain("configuration", e, ()))?;
        #[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
        {
            let result = execution::run(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&result).map_err(|e| Failure::retain(
                    "report",
                    e,
                    ()
                ))?
            );
            Ok(())
        }
        #[cfg(not(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4")))]
        {
            let _ = config;
            Err(Failure::retain(
                "feature",
                "requires cuda-cutlass-nvfp4,cuda-sage-vsa-fp4",
                (),
            ))
        }
    })();
    if let Err(failure) = result {
        std::mem::forget(failure.owner);
        eprintln!(
            "{}",
            serde_json::json!({"schema":"ferrum.fast_h3.dit.failure.v1","stage":failure.stage,"error":failure.message,
            "retirement":"not claimed; submitted owner chain retained until process exit"})
        );
        std::process::exit(1);
    }
}
