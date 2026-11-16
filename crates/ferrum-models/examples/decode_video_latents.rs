//! Original full-depth FastH3 latent -> RGB frame artifact consumer.
//! Run with `--features cuda-cutlass-nvfp4 -- --configuration FILE.json`.
//! The input is explicit little-endian F32 NCTHW; no prompt/DiT/audio is run.
#[path = "decode_video_latents/config.rs"]
mod config;
#[cfg(feature = "cuda-cutlass-nvfp4")]
#[path = "decode_video_latents/execution.rs"]
mod execution;
#[path = "support/retained_failure.rs"]
mod retained_failure;
use retained_failure::Failure;
fn main() {
    let result = (|| -> Result<(), Failure> {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() != Some("--configuration") {
            return Err(Failure::retain(
                "arguments",
                "usage: decode_video_latents --configuration FILE.json",
                (),
            ));
        }
        let path = args
            .next()
            .ok_or_else(|| Failure::retain("arguments", "missing configuration file", ()))?;
        if args.next().is_some() {
            return Err(Failure::retain("arguments", "unexpected argument", ()));
        }
        let c = config::Configuration::load(std::path::Path::new(&path))
            .map_err(|e| Failure::retain("configuration", e, ()))?;
        #[cfg(feature = "cuda-cutlass-nvfp4")]
        {
            let report = execution::run(c)?;
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
            let _ = c;
            Err(Failure::retain(
                "feature",
                "requires --features cuda-cutlass-nvfp4",
                (),
            ))
        }
    })();
    if let Err(f) = result {
        std::mem::forget(f.owner);
        eprintln!(
            "{}",
            serde_json::json!({"stage":f.stage,"error":f.message,"retirement":"not claimed; original owners retained until process exit"})
        );
        std::process::exit(1);
    }
}
