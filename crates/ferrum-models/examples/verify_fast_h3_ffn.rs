//! Managed FFN qualification, not video generation or a throughput benchmark.
//! `--model-dir PATH` checks a real selected block; `--fixture dyadic` runs an
//! independent exact-subset oracle and requires the test-support feature.
use serde_json::json;
#[path = "support/retained_failure.rs"]
mod retained_failure;
use retained_failure::Failure;

#[path = "verify_fast_h3_ffn/arguments.rs"]
mod arguments;
#[cfg(any(feature = "test-support", test))]
#[path = "verify_fast_h3_ffn/dyadic.rs"]
mod dyadic;
#[cfg(feature = "cuda-cutlass-nvfp4")]
#[path = "verify_fast_h3_ffn/execution.rs"]
mod execution;
#[cfg(any(feature = "cuda-cutlass-nvfp4", test))]
#[path = "support/fast_h3_package.rs"]
mod package;

fn main() {
    let result = (|| -> Result<(), Failure> {
        let Some(args) = arguments::Arguments::parse(std::env::args().skip(1))
            .map_err(|e| Failure::retain("arguments", e, ()))?
        else {
            println!("{}", arguments::USAGE);
            return Ok(());
        };
        #[cfg(feature = "cuda-cutlass-nvfp4")]
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
        #[cfg(not(feature = "cuda-cutlass-nvfp4"))]
        {
            let _ = args;
            Err(Failure::retain(
                "feature",
                "requires --features cuda-cutlass-nvfp4 (plus test-support for --fixture dyadic)",
                (),
            ))
        }
    })();
    if let Err(failure) = result {
        // Even a failed stderr write must not unwind through native owners.
        std::mem::forget(failure.owner);
        eprintln!(
            "{}",
            json!({"schema":"ferrum.fast_h3.ffn.qualification.failure.v1","stage":failure.stage,"error":failure.message,"retirement":"not claimed; original failure and owner chain retained until process exit"})
        );
        std::process::exit(1);
    }
}
